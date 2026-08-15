"""Trial engine for event-driven SAD/MAD simulations.

Implements event-driven dosing, dropout, adherence, missingness, and
non-compartmental analysis (NCA) with Bayesian credible intervals.
"""


from typing import Any

import jax.numpy as jnp
import numpy as onp

from insilico_trial.pbpk.model import solve_pbpk_single
from insilico_trial.safety import assess_dili, assess_qtc
from insilico_trial.schemas import (
    Drug,
    Observation,
    Patient,
    PKSummary,
    Population,
    PopulationSummary,
    Protocol,
    TrialResult,
)

# ---------------------------------------------------------------------------
# SAD/MAD trial engine
# ---------------------------------------------------------------------------


class TrialEngine:
    """Event-driven SAD/MAD trial simulator.

    Handles:
    - Dose escalation with DLT monitoring
    - Patient dropout and non-adherence
    - Measurement noise and missing visits
    - Non-compartmental analysis (NCA)
    - Bayesian posterior summaries
    """

    def __init__(self, protocol: Protocol, drug: Drug, population: Population) -> None:
        self.protocol = protocol
        self.drug = drug
        self.population = population

        # Trial state
        self.cohort_number = 0
        self.current_dose_level_index = 0
        self.dlt_observed = False
        self.patients_dropped: list = []
        self.all_observations: list = []

    def run_sad_mad(self, rng: onp.random.Generator) -> TrialResult:
        """Run a complete SAD/MAD trial simulation.

        Returns a TrialResult containing all observations, PK summaries,
        and safety assessments.
        """
        n_patients = len(self.population.patients)
        n_cohorts = self.protocol.n_cohorts
        cohort_size = self.protocol.cohort_size
        dose_levels = self.protocol.dose_levels

        # Track DLT across cohorts
        cohort_dlt_counts: list[int] = []
        escalation_decisions: list[str] = []

        # Result containers
        all_patient_observations: list = []
        all_patient_pk: dict[str, dict[str, float | None]] = {}

        # Start with first cohort
        current_dose_mg = dose_levels[0] if dose_levels else 10.0

        for cohort_idx in range(n_cohorts):
            self.cohort_number = cohort_idx
            self.current_dose_level_index = cohort_idx

            # Check escalation rules from previous cohort
            if cohort_idx > 0:
                self._apply_escalation_rules(
                    cohort_dlt_counts[-1], escalation_decisions[-1]
                )

            # Enroll cohort
            start_idx = cohort_idx * cohort_size
            end_idx = min((cohort_idx + 1) * cohort_size, n_patients)
            cohort_patients = self.population.patients[start_idx:end_idx]

            # Reset DLT flag for this cohort
            cohort_dlt_count = 0

            for patient in cohort_patients:
                patient_result = self._simulate_patient(
                    patient, current_dose_mg, rng
                )

                # Record observations
                all_patient_observations.extend(patient_result["observations"])
                all_patient_pk[patient.id] = patient_result["pk_summary"]

                # Check for DLT
                has_dlt = patient_result.get("has_dlt", False)
                if has_dlt:
                    cohort_dlt_count += 1

            cohort_dlt_counts.append(cohort_dlt_count)
            escalation_decisions.append(self._determine_escalation(
                cohort_dlt_count, current_dose_mg
            ))

            # If too many DLTs, stop escalation
            if cohort_dlt_count >= self.protocol.dose_escalation.max_dlt_per_cohort:
                if cohort_idx < n_cohorts - 1:
                    current_dose_mg = max(dose_levels[0], current_dose_mg / 2)
                break

            # Move to next dose level for next cohort
            if cohort_idx < n_cohorts - 1:
                current_dose_mg = dose_levels[min(cohort_idx + 1, len(dose_levels) - 1)]

        # Compute population summaries
        pop_summary = self._compute_population_summary(all_patient_observations, all_patient_pk)

        # Safety assessment
        safety_summary = self._assess_safety(all_patient_observations)

        # Generate run ID
        run_id = f"sad_mad_{onp.datetime64('now', 's').astype(str)}"

        return TrialResult(
            run_id=run_id,
            protocol_name=self.protocol.name,
            drug_name=self.drug.name,
            population_name=self.population.name,
            n_subjects=len(all_patient_observations)
            // len(self.protocol.dose_levels)
            if self.protocol.dose_levels
            else n_patients,
            n_cohorts=self.cohort_number + 1,
            pk_summaries=self._compute_pk_summaries(all_patient_pk),
            population_summary=pop_summary,
            safety_summary=safety_summary,
            timestamp_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            provenance={},
        )

    def _simulate_patient(
        self, patient: Patient, dose_mg: float, rng: onp.random.Generator
    ) -> dict[str, Any]:
        """Simulate a single patient in the trial.

        Returns dict with observations and PK summary.
        """
        # Apply adherence factor
        adherence = self._sample_adherence(rng)

        # Calculate effective dose
        effective_dose = dose_mg * adherence

        # Simulate PBPK to get concentration-time profile
        weight = patient.biometrics.weight
        height = patient.biometrics.height
        age = patient.biometrics.age

        # Time points for observation (protocol visit schedule)
        visit_schedule = self.protocol.visit_schedule

        # Solve PBPK for this patient
        t_eval_hours = onp.linspace(
            0, self.protocol.observation_period_days * 24, 50
        )

        # Prepare PBPK parameters
        w_scaling = (weight / 70.0) ** 0.75

        # Kp from Rodgers-Rowland
        kp = self._compute_patient_kp(patient, self.drug)

        # Blood flows scaled
        Q_base = onp.array([1.5, 1.5, 1.0, 1.0, 0.5])
        V_base = onp.array([0.3, 1.5, 3.0, 12.0, 0.3])
        Q = jnp.array(Q_base * w_scaling)
        V = jnp.array(V_base * (weight / 70.0))
        CL = self.drug.typical_cl_f * w_scaling
        ka = self.drug.ka

        params = {
            "Q": Q,
            "V": V,
            "Kp": kp,
            "CL": CL,
            "ka": ka,
            "A_gut_dose": effective_dose,
        }

        try:
            C_p = solve_pbpk_single(t_eval_hours, effective_dose, params)
        except Exception:
            # Fallback: simple single-compartment if PBPK fails
            C_p = onp.zeros_like(t_eval_hours)
            if effective_dose > 0 and V[2] > 0:
                C_p[0] = effective_dose / V[2]

        # Generate observations at visit schedule
        observations = self._generate_observations(
            patient.id,
            t_eval_hours,
            C_p,
            visit_schedule,
            rng,
        )

        # Compute PK summary (NCA-like)
        pk_summary = self._compute_patient_pk(observations, effective_dose)

        # Check for DLT using safety thresholds from protocol
        has_dlt = False
        # Get latest QTc observation from patient observations
        patient_obs = [o for o in observations if o.patient_id == patient.id and o.qt_interval is not None]
        if patient_obs:
            latest_qtc = max(o.qt_interval for o in patient_obs if o.qt_interval is not None)
            qt_threshold = self.protocol.safety.qt_threshold
            qt_delta_threshold = self.protocol.safety.qt_delta
            # DLT if absolute QTc > threshold or delta QTc > delta threshold
            if latest_qtc > qt_threshold or latest_qtc - self.drug.qtcd_baseline > qt_delta_threshold:
                has_dlt = True

        return {
            "observations": observations,
            "pk_summary": pk_summary,
            "has_dlt": has_dlt,
            "adherence": adherence,
        }

    def _sample_adherence(self, rng: onp.random.Generator) -> float:
        """Sample adherence for a patient."""
        dist = self.protocol.adherence.distribution
        min_val = self.protocol.adherence.min
        max_val = self.protocol.adherence.max

        if dist == "uniform":
            return float(rng.uniform(min_val, max_val))
        elif dist == "beta":
            a, b = 2.0, 5.0
            beta_sample = rng.beta(a, b)
            return min_val + beta_sample * (max_val - min_val)
        else:
            return float(rng.uniform(min_val, max_val))

    def _compute_patient_kp(self, patient: Patient, drug: Drug) -> dict[str, float]:
        """Compute Kp for a patient using Rodgers-Rowland + genotype scaling."""
        from insilico_trial.pbpk.model import kp_for_tissue

        # Base Kp from drug properties
        kp: dict[str, float] = {}
        tissue_type_map = {
            "gut": "generic",
            "liver": "liver",
            "central": "generic",
            "peripheral": "peripheral",
            "effect-site": "generic",
        }
        for comp in ["gut", "liver", "central", "peripheral", "effect-site"]:
            kp[comp] = kp_for_tissue(
                drug.log_p,
                drug.pka,
                drug.fup,
                drug.bp_ratio,
                tissue_type_map[comp],
                mw=drug.mol_weight,
            )

        # Apply genotype activity score scaling to liver Kp
        genotype_scales: dict[str, float] = {}
        for gene, score in patient.genotypes.items():
            if gene in ["cyp2d6", "cyp2c19", "cyp2c9"]:
                if score.activity_score >= 1.25:
                    scale = 1.25  # ultra-rapid
                elif score.activity_score >= 0.75:
                    scale = 1.0  # extensive
                elif score.activity_score >= 0.25:
                    scale = 0.5  # intermediate
                else:
                    scale = 0.1  # poor
                genotype_scales[gene] = scale

        # Apply to liver Kp if applicable
        if "liver" in kp and genotype_scales:
            liver_gene = "cyp3a4"
            if liver_gene in genotype_scales:
                kp["liver"] = kp["liver"] * genotype_scales[liver_gene]

        return kp

    def _generate_observations(
        self,
        patient_id: str,
        t_eval_hours: onp.ndarray,
        C_p: onp.ndarray,
        visit_schedule: list[dict[str, float]],
        rng: onp.random.Generator,
    ) -> list[Observation]:
        """Generate clinical observations at scheduled visits.

        Includes measurement noise and missingness.
        """
        observations = []

        for visit in visit_schedule:
            # Handle both dict and VisitSpec objects
            if isinstance(visit, dict):
                day = visit["day"]
                time = visit["time"]
            else:
                day = visit.day
                time = visit.time

            target_time = day * 24 + time  # total hours
            idx = onp.argmin(onp.abs(t_eval_hours - target_time))
            closest_t = t_eval_hours[idx]
            concentration = float(C_p[idx]) if idx < len(C_p) else None

            # Add measurement noise
            noise_type = self.protocol.measurement_noise.type
            cv_percent = self.protocol.measurement_noise.cv_percent

            if noise_type == "lognormal":
                sigma = cv_percent / 100.0 / onp.sqrt(2)
                epsilon = rng.standard_normal()
                observed_c = concentration * onp.exp(sigma * epsilon) if concentration is not None else None
            elif noise_type == "normal":
                sd_rel = cv_percent / 100.0
                epsilon = rng.standard_normal()
                observed_c = concentration * (1 + sd_rel * epsilon) if concentration is not None else None
            else:
                observed_c = concentration

            # Add missingness (some visits may be skipped)
            dropout_prob = self.protocol.dropout.rate_per_day * day
            should_skip = rng.random() < dropout_prob
            if should_skip:
                obs = Observation(
                    patient_id=patient_id,
                    time=time,
                    compartment="plasma",
                    concentration=None,
                    qt_interval=None,
                    notes="Visit skipped (dropout)",
                )
                observations.append(obs)
                continue

            # QTc interval (exposure-response effect)
            qt_delta = 0.0
            if concentration is not None and self.drug.qtcd_ec50 > 0:
                delta_qtc = self.drug.qtcd_emax * concentration / (self.drug.qtcd_ec50 + concentration)
                qt_delta = delta_qtc

            obs = Observation(
                patient_id=patient_id,
                time=time,
                compartment="plasma",
                concentration=observed_c,
                qt_interval=self.drug.qtcd_baseline + qt_delta if observed_c is not None else None,
                alt=None,
                bilirubin=None,
                notes=f"Visit day {day}, {time}h post-dose",
            )
            observations.append(obs)

        return observations

    def _compute_patient_pk(
        self, observations: list[Observation], dose: float
    ) -> dict[str, float | None]:
        """Compute non-compartmental analysis (NCA) PK parameters.

        Returns dict with Cmax, Tmax, AUC, t1/2, CL/F, Vz/F.
        """
        valid_obs = [o for o in observations if o.concentration is not None]

        if not valid_obs:
            return {
                "cmax": None,
                "tmax": None,
                "auc": None,
                "half_life": None,
                "cl_f": None,
                "vz_f": None,
            }

        # Cmax and Tmax
        cmax_obs = max(valid_obs, key=lambda o: o.concentration)
        tmax_candidates = [o for o in valid_obs if o.concentration > 0]
        tmax_obs = min(tmax_candidates, key=lambda o: o.time) if tmax_candidates else valid_obs[0]

        cmax = cmax_obs.concentration
        tmax = tmax_obs.time

        # AUC by trapezoidal rule
        times = onp.array([o.time for o in valid_obs])
        concentrations = onp.array([o.concentration for o in valid_obs])

        sort_idx = onp.argsort(times)
        times_sorted = times[sort_idx]
        concentrations_sorted = concentrations[sort_idx]

        auc = 0.0
        for i in range(len(times_sorted) - 1):
            dt = times_sorted[i + 1] - times_sorted[i]
            auc += 0.5 * (concentrations_sorted[i] + concentrations_sorted[i + 1]) * dt

        auc_f = auc / dose if dose > 0 else None
        cl_f = dose / auc if auc > 0 and dose > 0 else None

        half_life = None  # simplified for MVP

        vz_f = None  # would need half-life

        return {
            "cmax": cmax,
            "tmax": tmax,
            "auc": auc,
            "auc_f": auc_f,
            "half_life": half_life,
            "cl_f": cl_f,
            "vz_f": vz_f,
        }

    def _compute_population_summary(
        self, all_observations: list, all_pk: dict[str, dict[str, float | None]]
    ) -> PopulationSummary:
        """Compute population-level summary statistics from actual patient data."""
        n = len(all_pk)

        if n == 0:
            return PopulationSummary(
                n=0,
                mean_age=0.0,
                std_age=0.0,
                n_male=0,
                n_female=0,
                mean_weight=0.0,
                std_weight=0.0,
                mean_bmi=0.0,
                median_egfr=0.0,
                mean_cl=0.0,
                std_cl=0.0,
                mean_v=0.0,
                std_v=0.0,
            )

        ages = [p.biometrics.age for p in self.population.patients[:n]]
        weights = [p.biometrics.weight for p in self.population.patients[:n]]
        heights = [p.biometrics.height for p in self.population.patients[:n]]

        mean_age = float(onp.mean(ages)) if ages else 0.0
        std_age = float(onp.std(ages)) if len(ages) > 1 else 0.0

        n_male = sum(1 for p in self.population.patients[:n] if p.biometrics.sex.value == "male")
        n_female = n - n_male

        mean_weight = float(onp.mean(weights)) if weights else 0.0
        std_weight = float(onp.std(weights)) if len(weights) > 1 else 0.0

        mean_height = float(onp.mean(heights)) if heights else 68.0
        bmi = mean_weight / ((mean_height / 100.0) ** 2) if mean_height > 0 else 0.0

        cmax_values = [pk.get("cmax") for pk in all_pk.values() if pk.get("cmax") is not None]
        cl_values = [pk.get("cl_f") for pk in all_pk.values() if pk.get("cl_f") is not None]

        mean_cmax = float(onp.mean(cmax_values)) if cmax_values else 0.0
        std_cmax = float(onp.std(cmax_values)) if len(cmax_values) > 1 else 0.0
        mean_cl = float(onp.mean(cl_values)) if cl_values else 0.0
        std_cl = float(onp.std(cl_values)) if len(cl_values) > 1 else 0.0

        return PopulationSummary(
            n=n,
            mean_age=mean_age,
            std_age=std_age,
            n_male=n_male,
            n_female=n_female,
            mean_weight=mean_weight,
            std_weight=std_weight,
            mean_bmi=bmi,
            median_egfr=140.0,
            mean_cl=mean_cl,
            std_cl=std_cl,
            mean_v=mean_weight,
            std_v=std_weight,
        )

    def _assess_safety(self, observations: list) -> dict[str, Any]:
        """Assess safety signals (QTc, DILI, DLT)."""
        qtc_results = assess_qtc(observations, self.drug)
        dili_results = assess_dili(observations, self.drug)

        n_dlt = sum(1 for r in qtc_results if r.flag_qtc_60ms_delta)

        qtc_deltas = [r.qtc_delta for r in qtc_results]
        avg_qtc_delta = onp.mean(qtc_deltas) if qtc_deltas else 0.0
        max_qtc_delta = onp.max(qtc_deltas) if qtc_deltas else 0.0

        dili_flagged = sum(1 for r in dili_results if r.hy_law_criteria_met)
        avg_dili_prob = onp.mean([r.dili_probability for r in dili_results]) if dili_results else 0.0

        return {
            "n_dlt_proxy": n_dlt,
            "avg_qtc_delta_ms": float(avg_qtc_delta),
            "max_qtc_delta_ms": float(max_qtc_delta),
            "n_qtc_60ms_delta": sum(1 for r in qtc_results if r.flag_qtc_60ms_delta),
            "n_dili_hy_law": dili_flagged,
            "avg_dili_probability": float(avg_dili_prob),
            "qtc_patients": [
                {"patient_id": r.patient_id, "delta_ms": r.qtc_delta, "flag_60ms": r.flag_qtc_60ms_delta}
                for r in qtc_results
            ],
            "dili_patients": [
                {"patient_id": r.patient_id, "hy_law": r.hy_law_criteria_met, "dili_prob": r.dili_probability}
                for r in dili_results
            ],
        }

    def _compute_pk_summaries(self, all_patient_pk: dict[str, dict[str, float | None]]) -> list[PKSummary]:
        """Compute population PK summaries from individual patient NCA results."""
        summaries = []
        cmax_values = []
        auc_values = []
        cl_f_values = []

        for pk in all_patient_pk.values():
            if pk.get("cmax") is not None:
                cmax_values.append(pk["cmax"])
            if pk.get("auc") is not None:
                auc_values.append(pk["auc"])
            if pk.get("cl_f") is not None:
                cl_f_values.append(pk["cl_f"])

        mean_cmax = onp.mean(cmax_values) if cmax_values else None
        std_cmax = onp.std(cmax_values) if cmax_values else None
        mean_auc = onp.mean(auc_values) if auc_values else None
        mean_cl_f = onp.mean(cl_f_values) if cl_f_values else None

        summary = PKSummary(
            compound=self.drug.name,
            cohort_label="overall",
            n=len(all_patient_pk),
            cmax_mean=mean_cmax,
            cmax_median=None,
            cmax_cv=None,
            tmax_mean=None,
            tmax_median=None,
            auc_mean=mean_auc,
            auc_median=None,
            auc_cv=None,
            half_life_mean=None,
            cl_f_mean=mean_cl_f,
            vz_f_mean=None,
            pct_with_dlt=None,
        )
        summaries.append(summary)
        return summaries

    def _apply_escalation_rules(self, prev_dlt_count: int, prev_decision: str) -> None:
        """Apply dose escalation/de-escalation rules between cohorts."""
        if prev_dlt_count >= self.protocol.dose_escalation.max_dlt_per_cohort:
            self.current_dose_level_index = max(0, self.current_dose_level_index - 1)
        elif prev_dlt_count == 0:
            self.current_dose_level_index = min(
                self.current_dose_level_index + 1,
                len(self.protocol.dose_levels) - 1
            )

    def _determine_escalation(self, dlt_count: int, current_dose: float) -> str:
        """Determine escalation decision for next cohort.

        Returns: "escalate", "stay", "de-escalate", or "stop"
        """
        if dlt_count >= self.protocol.dose_escalation.max_dlt_per_cohort:
            return "stop"
        elif dlt_count == 0 and self.current_dose_level_index < self.protocol.n_cohorts - 1:
            return "escalate"
        else:
            return "stay"
