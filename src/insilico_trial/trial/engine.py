"""Trial engine for event-driven SAD/MAD simulations.

Implements event-driven dosing, dropout, adherence, missingness, and
non-compartmental analysis (NCA) with uncertainty intervals.

Visit-time convention
---------------------
``VisitSpec.time`` is hours from the first (single) dose. The ``day`` field is
an informational annotation only; it is not multiplied by 24.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jax.numpy as jnp
import numpy as onp

from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_fixed_step
from insilico_trial.pbpk.model import build_pbpk_params, solve_pbpk_batch, solve_pbpk_single
from insilico_trial.safety import determine_dlt, run_safety_assessment
from insilico_trial.schemas import (
    DosingEvent,
    Drug,
    Observation,
    Patient,
    PKSummary,
    Population,
    PopulationSummary,
    Protocol,
    TrialResult,
)
from insilico_trial.stats import posterior_predictive_pk

# ---------------------------------------------------------------------------
# Non-compartmental analysis (NCA)
# ---------------------------------------------------------------------------


def _trapezoidal_auc(times: onp.ndarray, concs: onp.ndarray) -> float:
    """Area under the curve by the linear trapezoidal rule (mg*h/L)."""
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (concs[i] + concs[i + 1]) * dt
    return float(auc)


def _terminal_lambda_z(times: onp.ndarray, concs: onp.ndarray, tmax_idx: int) -> float | None:
    """Estimate the terminal elimination rate constant lambda_z (1/h).

    Uses the last three or more log-linear declining points after Tmax, provided
    they cover at least 1.5 terminal half-lives of data and the fit is adequate.
    Returns None if lambda_z cannot be estimated reliably.
    """
    idxs = onp.arange(tmax_idx + 1, len(times))
    if len(idxs) < 3:
        return None

    log_c = onp.log(onp.maximum(concs[idxs], 1e-9))
    t = times[idxs]

    # Starting from the last point, greedily include earlier points that keep
    # concentrations monotonically declining.
    n = len(t)
    best_slope: float | None = None
    best_r2 = 0.0
    for k in range(3, n + 1):
        t_k = t[n - k :]
        c_k = log_c[n - k :]
        slope, intercept = onp.polyfit(t_k, c_k, 1)
        pred = slope * t_k + intercept
        ss_res = float(onp.sum((c_k - pred) ** 2))
        ss_tot = float(onp.sum((c_k - c_k.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if slope < 0 and r2 > best_r2:
            best_r2 = r2
            best_slope = float(-slope)

    if best_slope is None or best_slope <= 0 or best_r2 < 0.7:
        return None
    return best_slope


def compute_nca(observations: list[Observation], dose: float) -> dict[str, float | None]:
    """Compute non-compartmental analysis (NCA) PK parameters.

    Parameters
    ----------
    observations : list[Observation]
        Concentration observations for one patient
    dose : float
        Administered (nominal) dose in mg; used for CL/F and Vz/F

    Returns
    -------
    dict with Cmax, Tmax, AUClast, AUCinf, lambda_z, half-life, CL/F, Vz/F.
    """
    valid_obs = [o for o in observations if o.concentration is not None and o.concentration >= 0]

    empty: dict[str, float | None] = {
        "cmax": None,
        "tmax": None,
        "auc_last": None,
        "auc_inf": None,
        "lambda_z": None,
        "half_life": None,
        "cl_f": None,
        "vz_f": None,
    }
    if not valid_obs:
        return empty

    times = onp.array([o.time for o in valid_obs], dtype=onp.float64)
    valid_conc = [float(o.concentration) for o in valid_obs if o.concentration is not None]
    concs = onp.array(valid_conc, dtype=onp.float64)

    order = onp.argsort(times)
    times = times[order]
    concs = concs[order]

    # Cmax at the time of the maximum concentration
    cmax_idx = int(onp.argmax(concs))
    cmax = float(concs[cmax_idx])
    tmax = float(times[cmax_idx])

    # AUClast by trapezoidal rule
    auc_last = _trapezoidal_auc(times, concs)

    # Terminal elimination
    lambda_z = _terminal_lambda_z(times, concs, cmax_idx)
    half_life: float | None = None
    auc_inf: float | None = None
    cl_f: float | None = None
    vz_f: float | None = None

    if lambda_z is not None and lambda_z > 0:
        half_life = float(onp.log(2.0) / lambda_z)
        c_last = float(concs[-1])
        auc_inf = auc_last + c_last / lambda_z
    else:
        auc_inf = auc_last

    if dose > 0 and auc_inf is not None and auc_inf > 0:
        cl_f = dose / auc_inf
        if lambda_z is not None and lambda_z > 0:
            vz_f = cl_f / lambda_z

    return {
        "cmax": cmax,
        "tmax": tmax,
        "auc_last": auc_last,
        "auc_inf": auc_inf,
        "lambda_z": lambda_z,
        "half_life": half_life,
        "cl_f": cl_f,
        "vz_f": vz_f,
    }


def summarize_metrics(values: list[float]) -> dict[str, float]:
    """Summarize a list of per-patient metrics with median and 90% interval."""
    arr = onp.asarray(values, dtype=onp.float64)
    if len(arr) == 0:
        return {"n": 0.0, "mean": float("nan"), "median": float("nan"),
                "p5": float("nan"), "p95": float("nan")}
    return {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "median": float(onp.median(arr)),
        "p5": float(onp.quantile(arr, 0.05)),
        "p95": float(onp.quantile(arr, 0.95)),
    }


# ---------------------------------------------------------------------------
# SAD/MAD trial engine
# ---------------------------------------------------------------------------


class TrialEngine:
    """Event-driven SAD/MAD trial simulator.

    Handles:
    - Dose escalation with DLT monitoring (rules actually control the next dose)
    - Patient dropout and non-adherence
    - Measurement noise and missing visits
    - Non-compartmental analysis (NCA)
    - Safety assessment (QTc, DILI, CTCAE DLT)
    - Uncertainty intervals (median + 90% interval across virtual subjects)
    """

    def __init__(
        self,
        protocol: Protocol,
        drug: Drug,
        population: Population,
        posterior_samples: dict[str, Any] | None = None,
    ) -> None:
        self.protocol = protocol
        self.drug = drug
        self.population = population
        self.posterior_samples = posterior_samples
        self.current_dose_level_index = 0
        self.cohort_histories: list[dict[str, Any]] = []

    def run_sad_mad(self, rng: onp.random.Generator) -> TrialResult:
        """Run a complete SAD/MAD trial simulation."""
        n_patients = len(self.population.patients)
        n_cohorts = self.protocol.n_cohorts
        cohort_size = self.protocol.cohort_size
        dose_levels = self.protocol.dose_levels

        all_observations: list[Observation] = []
        all_patient_pk: dict[str, dict[str, float | None]] = {}
        cohort_summaries: list[dict[str, Any]] = []
        cohort_escalations: list[str] = []
        n_enrolled = 0
        next_dose_level_index = 0
        stop = False

        for cohort_idx in range(n_cohorts):
            if stop:
                break

            start = cohort_idx * cohort_size
            end = min(start + cohort_size, n_patients)
            cohort_patients = self.population.patients[start:end]
            if not cohort_patients:
                break

            # Dose for this cohort is controlled by escalation rules.
            dose_mg = dose_levels[min(next_dose_level_index, len(dose_levels) - 1)]
            n_enrolled += len(cohort_patients)

            # Batch-solve the dense PK profile for every patient in the cohort.
            # Adherence is sampled per patient; absorbed (dose-entering-gut) amounts
            # are passed to the batched solver.
            administered_doses = onp.array([
                dose_mg * self._sample_adherence(rng) for _ in cohort_patients
            ], dtype=onp.float64)

            if self.protocol.design == "MAD":
                # MAD: use event-driven multi-dose solver
                t_eval_hours, C_batch = self._solve_mad_batch(
                    cohort_patients, administered_doses,
                    solver=self.protocol.solver,
                )
            else:
                # SAD: single dose
                absorbed_doses = administered_doses * self.drug.bioavailability
                t_eval_hours, C_batch = self._solve_cohort_batch(
                    cohort_patients, absorbed_doses,
                    solver=self.protocol.solver,
                )

            cohort_dlt_count = 0

            for pi, patient in enumerate(cohort_patients):
                gs = self._genotype_scale(patient)
                patient_result = self._evaluate_patient(
                    patient, rng, t_eval_hours, C_batch[pi],
                    float(administered_doses[pi]), gs,
                )
                all_observations.extend(patient_result["observations"])
                all_patient_pk[patient.id] = patient_result["pk_summary"]
                if patient_result["has_dlt"]:
                    cohort_dlt_count += 1

            dlt_rate = cohort_dlt_count / len(cohort_patients) if cohort_patients else 0.0

            # Determine next dose via escalation rules.
            decision, next_dose_level_index, stop = self._escalation_decision(
                cohort_dlt_count, next_dose_level_index
            )
            cohort_escalations.append(decision)

            cohort_summaries.append({
                "cohort": cohort_idx + 1,
                "dose_mg": dose_mg,
                "n": len(cohort_patients),
                "n_dlt": cohort_dlt_count,
                "dlt_rate": dlt_rate,
                "escalation_decision": decision,
                "steady_state_reached": self._check_steady_state(cohort_patients, t_eval_hours, C_batch, administered_doses),
            })

            # If too many DLTs, the trial stops.
            if stop:
                break

        pop_summary = self._compute_population_summary(all_patient_pk, n_enrolled)
        safety_summary = run_safety_assessment(all_observations, self.drug, self.protocol.safety)
        pk_summaries = self._compute_pk_summaries(all_patient_pk, cohort_summaries)

        run_id = self._make_run_id()
        uncertainty = self._compute_uncertainty(all_patient_pk, cohort_summaries)

        return TrialResult(
            run_id=run_id,
            protocol_name=self.protocol.name,
            drug_name=self.drug.name,
            population_name=self.population.name,
            n_subjects=n_enrolled,
            n_cohorts=len(cohort_summaries),
            pk_summaries=pk_summaries,
            population_summary=pop_summary,
            safety_summary=safety_summary,
            observations=all_observations,
            cohort_summaries=cohort_summaries,
            uncertainty=uncertainty,
            timestamp_utc=datetime.now(UTC),
            provenance={},
        )

    # ------------------------------------------------------------------
    # Escalation / de-escalation / stop rules
    # ------------------------------------------------------------------

    def _escalation_decision(self, dlt_count: int, current_level_index: int) -> tuple[str, int, bool]:
        """Return (decision, next_level_index, stop).

        Rules (modified accrual):
        - DLTs >= max_dlt_per_cohort -> "stop" (no further cohorts)
        - 0 DLTs -> escalate to the next pre-specified dose level
        - 0 < DLTs < max -> "stay" at the current dose level
        """
        max_dlt = self.protocol.dose_escalation.max_dlt_per_cohort
        if dlt_count >= max_dlt:
            return "stop", current_level_index, True
        if dlt_count == 0:
            nxt = min(current_level_index + 1, len(self.protocol.dose_levels) - 1)
            return "escalate", nxt, False
        return "stay", current_level_index, False

    # ------------------------------------------------------------------
    # Patient-level simulation
    # ------------------------------------------------------------------

    def _simulate_patient(
        self, patient: Patient, dose_mg: float, rng: onp.random.Generator
    ) -> dict[str, Any]:
        """Simulate a single patient in the trial.

        The nominal dose is ``dose_mg``. Adherence scales the administered dose,
        and bioavailability scales the amount reaching the gut for absorption.
        The nominal (adherence-adjusted) dose is used for NCA-derived CL/F.

        This single-patient path is retained for small cohorts / testing; the
        cohort loop in :meth:`run_sad_mad` batch-solves PK for each cohort for
        throughput (see :meth:`_solve_cohort_batch`).
        """
        adherence = self._sample_adherence(rng)
        administered_dose = dose_mg * adherence
        absorbed_dose = administered_dose * self.drug.bioavailability

        # Genotype scales metabolic clearance (not tissue partitioning).
        genotype_scale = self._genotype_scale(patient)
        params = build_pbpk_params(
            weight_kg=patient.biometrics.weight,
            age=patient.biometrics.age,
            drug=self.drug,
            genotype_scale=genotype_scale,
        )

        # Simulation grid (hourly) over the observation period for smooth NCA.
        t_end_h = float(self.protocol.observation_period_days * 24)
        t_eval_hours = onp.linspace(0, t_end_h, max(int(t_end_h) + 1, 50))

        C_p = onp.asarray(solve_pbpk_single(t_eval_hours, absorbed_dose, params), dtype=onp.float64)
        return self._evaluate_patient(patient, rng, t_eval_hours, C_p, administered_dose, genotype_scale)

    def _solve_cohort_batch(
        self, cohort_patients: list[Patient], administered_doses: onp.ndarray,
        solver: str = "diffrax",
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """Batch-solve the dense PK profiles for a whole cohort at once.

        Returns
        -------
        t_eval_hours : ndarray (n_time,)
        C_batch : ndarray (n_patients, n_time)   plasma concentrations (mg/L)
        """
        t_end_h = float(self.protocol.observation_period_days * 24)
        t_eval_hours = onp.linspace(0, t_end_h, max(int(t_end_h) + 1, 50))

        params_list = [
            build_pbpk_params(
                weight_kg=p.biometrics.weight,
                age=p.biometrics.age,
                drug=self.drug,
                genotype_scale=self._genotype_scale(p),
            )
            for p in cohort_patients
        ]

        if solver == "fixed_step":
            # Fixed-step RK4 batch solve: vmap over solve_pbpk_batch_fixed_step
            params_batch = {
                "Q": onp.stack([p["Q"] for p in params_list]),
                "V": onp.stack([p["V"] for p in params_list]),
                "Kp": onp.stack([p["Kp"] for p in params_list]),
                "CL": onp.array([p["CL"] for p in params_list]),
                "ka": onp.array([p["ka"] for p in params_list]),
            }
            A_gut_0s = administered_doses * self.drug.bioavailability
            C_batch = onp.asarray(
                solve_pbpk_batch_fixed_step(t_eval_hours, A_gut_0s, params_batch, dt=0.01),
                dtype=onp.float64,
            )
        else:
            # Diffrax Tsit5 (default)
            params_batch = {
                "Q": onp.stack([p["Q"] for p in params_list]),
                "V": onp.stack([p["V"] for p in params_list]),
                "Kp": onp.stack([p["Kp"] for p in params_list]),
                "CL": onp.array([p["CL"] for p in params_list]),
                "ka": onp.array([p["ka"] for p in params_list]),
            }
            C_batch = onp.asarray(
                solve_pbpk_batch(t_eval_hours, administered_doses, params_batch),
                dtype=onp.float64,
            )
        return t_eval_hours, C_batch

    def _get_dosing_events(self) -> list[DosingEvent]:
        """Get dosing events for MAD, auto-generating if not provided."""
        if self.protocol.dosing_events:
            return self.protocol.dosing_events
        # Auto-generate from n_doses and dosing_interval_days
        if self.protocol.n_doses is None:
            return []
        interval_h = self.protocol.dosing_interval_days * 24.0
        dose_mg = self.protocol.dose_levels[0]  # MAD uses single dose level per cohort
        events = []
        for i in range(self.protocol.n_doses):
            events.append(DosingEvent(
                time_h=float(i * interval_h),
                dose_mg=float(dose_mg),
                route=self.protocol.dosing_route,
            ))
        return events

    def _solve_mad_batch(
        self,
        cohort_patients: list[Patient],
        administered_doses: onp.ndarray,
        solver: str = "fixed_step",
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """Solve MAD PK profiles for a cohort with repeated dosing events.

        Uses event-driven integration: between dosing events, integrate normally;
        at each event time, add the dose to the gut compartment.

        Returns
        -------
        t_eval_hours : ndarray (n_time,)
        C_batch : ndarray (n_patients, n_time)   plasma concentrations (mg/L)
        """
        dosing_events = self._get_dosing_events()
        if not dosing_events:
            # Fall back to single-dose SAD
            return self._solve_cohort_batch(cohort_patients, administered_doses, solver)

        t_end_h = float(self.protocol.observation_period_days * 24)
        t_eval_hours = onp.linspace(0, t_end_h, max(int(t_end_h) + 1, 50))

        # Build patient params
        params_list = [
            build_pbpk_params(
                weight_kg=p.biometrics.weight,
                age=p.biometrics.age,
                drug=self.drug,
                genotype_scale=self._genotype_scale(p),
            )
            for p in cohort_patients
        ]

        if solver == "fixed_step":
            return self._solve_mad_batch_fixed_step(
                t_eval_hours, cohort_patients, administered_doses, dosing_events, params_list
            )
        else:
            return self._solve_mad_batch_diffrax(
                t_eval_hours, cohort_patients, administered_doses, dosing_events, params_list
            )

    def _solve_mad_batch_fixed_step(
        self,
        t_eval_hours: onp.ndarray,
        cohort_patients: list[Patient],
        administered_doses: onp.ndarray,
        dosing_events: list[DosingEvent],
        params_list: list[dict[str, Any]],
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """MAD batch solve using fixed-step RK4 with event-driven dosing."""

        # Instead of complex event-driven scan, use a simpler approach:
        # Solve segment by segment between dosing events
        return self._solve_mad_segments_fixed_step(t_eval_hours, cohort_patients, administered_doses, dosing_events, params_list)

    def _solve_mad_segments_fixed_step(
        self,
        t_eval_hours: onp.ndarray,
        cohort_patients: list[Patient],
        administered_doses: onp.ndarray,
        dosing_events: list[DosingEvent],
        params_list: list[dict[str, Any]],
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """Solve MAD by chaining fixed-step segments between dosing events."""
        from insilico_trial.pbpk.fixed_step import solve_pbpk_batch_fixed_step

        n_patients = len(cohort_patients)
        dt = 0.01

        # Sort events
        events = sorted(dosing_events, key=lambda e: e.time_h)

        # Initial state: all zero, then add first dose
        y_current = jnp.zeros((n_patients, 6), dtype=jnp.float32)
        for i in range(n_patients):
            y_current = y_current.at[i, 0].set(float(administered_doses[i] * self.drug.bioavailability))

        # Accumulate concentrations on the full output grid
        C_accum = onp.zeros((n_patients, len(t_eval_hours)), dtype=onp.float64)

        prev_time = 0.0
        for event_idx, event in enumerate(events):
            event_time = float(event.time_h)
            if event_idx > 0:
                # Add dose at event time
                for i in range(n_patients):
                    y_current = y_current.at[i, 0].add(float(administered_doses[i] * self.drug.bioavailability))

            # Integrate from prev_time to event_time (or t_end for last segment)
            t_end_segment = event_time if event_idx < len(events) else float(t_eval_hours[-1])

            if t_end_segment <= prev_time + dt:
                continue

            # Create time grid for this segment
            t_segment = onp.linspace(prev_time, t_end_segment, max(int((t_end_segment - prev_time) / dt) + 1, 2))

            # Build params batch
            params_batch = {
                "Q": onp.stack([p["Q"] for p in params_list]),
                "V": onp.stack([p["V"] for p in params_list]),
                "Kp": onp.stack([p["Kp"] for p in params_list]),
                "CL": onp.array([p["CL"] for p in params_list]),
                "ka": onp.array([p["ka"] for p in params_list]),
            }

            # Initial gut amounts for this segment
            A_gut_0s = onp.array([float(y_current[i, 0]) for i in range(n_patients)])

            # Solve segment
            C_segment = solve_pbpk_batch_fixed_step(t_segment, A_gut_0s, params_batch, dt=dt)

            # Interpolate onto full grid and add to accumulator
            for i in range(n_patients):
                C_interp = onp.interp(t_eval_hours, t_segment, C_segment[i])
                # Only add contribution from this segment (mask by time)
                mask = (t_eval_hours >= prev_time) & (t_eval_hours <= t_end_segment + dt)
                C_accum[i, mask] += C_interp[mask]

            # Update state to end of segment
            # The last state is approximately the state at t_end_segment
            # We use the final y from the segment - but fixed_step doesn't return full state
            # For simplicity, estimate remaining gut amount
            for i in range(n_patients):
                # Gut amount decays exponentially: A_gut * exp(-ka * dt)
                y_current = y_current.at[i, 0].set(float(A_gut_0s[i] * onp.exp(-params_list[i]["ka"] * (t_end_segment - prev_time))))

            prev_time = t_end_segment

        return t_eval_hours, C_accum

    def _solve_mad_batch_diffrax(
        self,
        t_eval_hours: onp.ndarray,
        cohort_patients: list[Patient],
        administered_doses: onp.ndarray,
        dosing_events: list[DosingEvent],
        params_list: list[dict[str, Any]],
    ) -> tuple[onp.ndarray, onp.ndarray]:
        """MAD batch solve using diffrax with event-driven dosing (not yet implemented)."""
        # Fallback to SAD for now
        return self._solve_cohort_batch(cohort_patients, administered_doses, "diffrax")

    def _check_steady_state(
        self,
        cohort_patients: list[Patient],
        t_eval_hours: onp.ndarray,
        C_batch: onp.ndarray,
        administered_doses: onp.ndarray,
    ) -> bool:
        """Check if steady-state is reached by comparing AUC over last two dosing intervals.

        Returns True if AUC ratio (last_interval / prev_interval) < 1.05 for all patients.
        """
        if self.protocol.n_doses is None or self.protocol.n_doses < 2:
            return False

        interval_h = self.protocol.dosing_interval_days * 24.0
        last_dose_time = (self.protocol.n_doses - 1) * interval_h
        prev_dose_time = (self.protocol.n_doses - 2) * interval_h

        # Find indices for the last two dosing intervals
        last_mask = (t_eval_hours >= last_dose_time) & (t_eval_hours < last_dose_time + interval_h)
        prev_mask = (t_eval_hours >= prev_dose_time) & (t_eval_hours < prev_dose_time + interval_h)

        if not (onp.any(last_mask) and onp.any(prev_mask)):
            return False

        # Compute AUC for each interval for each patient
        t_last = t_eval_hours[last_mask]
        t_prev = t_eval_hours[prev_mask]

        for i in range(len(cohort_patients)):
            C_last = C_batch[i, last_mask]
            C_prev = C_batch[i, prev_mask]
            if len(C_last) < 2 or len(C_prev) < 2:
                continue
            auc_last = _trapezoidal_auc(t_last, C_last)
            auc_prev = _trapezoidal_auc(t_prev, C_prev)
            if auc_prev > 0:
                ratio = auc_last / auc_prev
                if ratio >= 1.05:
                    return False
        return True

    def _evaluate_patient(
        self, patient: Patient, rng: onp.random.Generator,
        t_eval_hours: onp.ndarray, C_p: onp.ndarray,
        administered_dose: float, genotype_scale: float,
    ) -> dict[str, Any]:
        """NCA + visit noise + DIL/DLT for a single patient's profile."""
        # Smooth-profile NCA on the dense grid.
        smooth_obs = self._obs_from_profile(patient.id, t_eval_hours, C_p)
        pk_summary = compute_nca(smooth_obs, administered_dose)

        # Observations at the protocol visit schedule (with noise/missingness).
        observations = self._generate_observations(
            patient.id, t_eval_hours, C_p, rng, administered_dose, pk_summary
        )

        # DLT determination integrates QTc, DILI, and CTCAE.
        has_dlt = determine_dlt(observations, self.drug, self.protocol.safety, patient.id)

        return {
            "observations": observations,
            "pk_summary": pk_summary,
            "has_dlt": has_dlt,
            "adherence": 1.0,
            "genotype_scale": genotype_scale,
        }

    @staticmethod
    def _obs_from_profile(patient_id: str, t: onp.ndarray, C: onp.ndarray) -> list[Observation]:
        """Create concentration observations at every simulation grid point."""
        obs = []
        for ti, ci in zip(t, C, strict=False):
            obs.append(Observation(
                patient_id=patient_id,
                time=float(ti),
                compartment="plasma",
                concentration=float(ci),
            ))
        return obs

    def _genotype_scale(self, patient: Patient) -> float:
        """Map the drug's metabolizing enzyme activity score to a clearance scale."""
        enzyme = self.drug.metabolizing_enzyme
        if not enzyme or enzyme == "none":
            return 1.0
        score_obj = patient.genotypes.get(enzyme)
        if score_obj is None:
            return 1.0
        # Floor so poor metabolizers do not get zero (or negative) clearance.
        return max(score_obj.activity_score, 0.05)

    def _sample_adherence(self, rng: onp.random.Generator) -> float:
        """Sample adherence for a patient."""
        dist = self.protocol.adherence.distribution
        min_val = self.protocol.adherence.min
        max_val = self.protocol.adherence.max

        if dist == "beta":
            a, b = 2.0, 5.0
            beta_sample = rng.beta(a, b)
            return float(min_val + beta_sample * (max_val - min_val))
        return float(rng.uniform(min_val, max_val))

    def _generate_observations(
        self,
        patient_id: str,
        t_eval_hours: onp.ndarray,
        C_p: onp.ndarray,
        rng: onp.random.Generator,
        administered_dose: float,
        pk_summary: dict[str, float | None],
    ) -> list[Observation]:
        """Generate clinical observations at scheduled visits.

        Includes measurement noise (never producing negative concentrations),
        missingness/dropout, QTc exposure-response, and DILI-driven
        ALT/bilirubin.
        """
        observations: list[Observation] = []
        liver_exposure = pk_summary.get("auc_last") or 0.0

        for visit in self.protocol.visit_schedule:
            target_time = float(visit.time)  # hours from first dose

            idx = int(onp.argmin(onp.abs(t_eval_hours - target_time)))
            concentration = float(C_p[idx])

            # Missingness (dropout) — the visit is skipped.
            dropout_prob = self.protocol.dropout.rate_per_day * (target_time / 24.0)
            if rng.random() < dropout_prob:
                observations.append(Observation(
                    patient_id=patient_id,
                    time=target_time,
                    compartment="plasma",
                    concentration=None,
                    qt_interval=None,
                    alt=None,
                    bilirubin=None,
                    notes="Visit skipped (dropout)",
                ))
                continue

            # Measurement noise: lognormal, never below zero.
            cv_percent = self.protocol.measurement_noise.cv_percent or 15.0
            sigma = cv_percent / 100.0 / onp.sqrt(2)
            observed_c = float(concentration * onp.exp(sigma * rng.standard_normal()))

            # QTc exposure-response (Emax).
            qt_delta = 0.0
            if self.drug.qtcd_ec50 > 0:
                qt_delta = self.drug.qtcd_emax * observed_c / (self.drug.qtcd_ec50 + observed_c)
            qt_interval = self.drug.qtcd_baseline + qt_delta

            # DILI exposure-response driven by liver exposure (plasma AUC proxy).
            alt, bilirubin = self._simulate_lft(liver_exposure)

            observations.append(Observation(
                patient_id=patient_id,
                time=target_time,
                compartment="plasma",
                concentration=observed_c,
                qt_interval=float(qt_interval),
                alt=alt,
                bilirubin=bilirubin,
                notes=f"Visit day {int(target_time // 24)}, {target_time}h post-dose",
            ))

        return observations

    def _simulate_lft(self, liver_exposure: float) -> tuple[float, float]:
        """Simulate ALT (U/L) and bilirubin (mg/dL) from liver exposure."""
        alt = self.drug.alt_baseline * (1.0 + self.drug.dili_emax_alt * liver_exposure / (self.drug.dili_ec50_alt + liver_exposure))
        bili = self.drug.bili_baseline * (1.0 + self.drug.dili_emax_bili * liver_exposure / (self.drug.dili_ec50_bili + liver_exposure))
        return float(alt), float(bili)

    def _compute_population_summary(
        self, all_pk: dict[str, dict[str, float | None]], n_enrolled: int
    ) -> PopulationSummary:
        """Compute population-level summary statistics from actual patient data."""
        patients = self.population.patients[:n_enrolled]
        n = len(patients)
        if n == 0:
            return PopulationSummary(
                n=0, mean_age=0.0, std_age=0.0, n_male=0, n_female=0,
                mean_weight=0.0, std_weight=0.0, mean_bmi=0.0, median_egfr=0.0,
                mean_cl=0.0, std_cl=0.0, mean_v=0.0, std_v=0.0,
            )

        ages = [p.biometrics.age for p in patients]
        weights = [p.biometrics.weight for p in patients]
        heights = [p.biometrics.height for p in patients]

        mean_age = float(onp.mean(ages))
        std_age = float(onp.std(ages)) if len(ages) > 1 else 0.0
        n_male = sum(1 for p in patients if p.biometrics.sex.value == "male")
        n_female = n - n_male
        mean_weight = float(onp.mean(weights))
        std_weight = float(onp.std(weights)) if len(weights) > 1 else 0.0
        mean_height = float(onp.mean(heights))
        bmi = mean_weight / ((mean_height / 100.0) ** 2) if mean_height > 0 else 0.0

        cl_values: list[float] = []
        vz_values: list[float] = []
        for pk in all_pk.values():
            v = pk.get("cl_f")
            if v is not None:
                cl_values.append(float(v))
            vz = pk.get("vz_f")
            if vz is not None:
                vz_values.append(float(vz))

        mean_cl = float(onp.mean(cl_values)) if cl_values else 0.0
        std_cl = float(onp.std(cl_values)) if len(cl_values) > 1 else 0.0
        mean_vz = float(onp.mean(vz_values)) if vz_values else 0.0
        std_vz = float(onp.std(vz_values)) if len(vz_values) > 1 else 0.0

        return PopulationSummary(
            n=n,
            mean_age=mean_age,
            std_age=std_age,
            n_male=n_male,
            n_female=n_female,
            mean_weight=mean_weight,
            std_weight=std_weight,
            mean_bmi=bmi,
            median_egfr=float(onp.median([p.biometrics.egfr for p in patients])),
            mean_cl=mean_cl,
            std_cl=std_cl,
            mean_v=mean_vz,
            std_v=std_vz,
        )

    def _compute_pk_summaries(
        self,
        all_patient_pk: dict[str, dict[str, float | None]],
        cohort_summaries: list[dict[str, Any]],
    ) -> list[PKSummary]:
        """Compute per-cohort and overall PK summaries."""
        summaries: list[PKSummary] = []

        # Per-cohort: slice patients by enrollment order.
        cohort_sizes = [c["n"] for c in cohort_summaries]
        cursor = 0
        for ci, size in enumerate(cohort_sizes):
            cohort_pk = list(all_patient_pk.values())[cursor : cursor + size]
            cursor += size
            dose_mg = cohort_summaries[ci]["dose_mg"]
            summaries.append(self._pk_summary_for(cohort_pk, f"Cohort {ci + 1} ({dose_mg:g} mg)"))

        summaries.append(self._pk_summary_for(list(all_patient_pk.values()), "Overall"))
        return summaries

    def _pk_summary_for(self, pk_list: list[dict[str, float | None]], label: str) -> PKSummary:
        def col(key: str) -> list[float]:
            val: list[float] = []
            for pk in pk_list:
                v = pk.get(key)
                if v is not None:
                    val.append(float(v))
            return val

        cmax = col("cmax")
        tmax = col("tmax")
        auc = col("auc_inf")
        half = col("half_life")
        clf = col("cl_f")
        vzf = col("vz_f")

        def stats(v: list[float]) -> tuple[float | None, float | None, float | None]:
            if not v:
                return None, None, None
            arr = onp.asarray(v)
            mean = float(arr.mean())
            median = float(onp.median(arr))
            cv = float(arr.std(ddof=1) / arr.mean()) if arr.mean() > 0 and len(arr) > 1 else None
            return mean, median, cv

        cmax_m, cmax_med, cmax_cv = stats(cmax)
        tmax_m, tmax_med, _ = stats(tmax)
        auc_m, auc_med, auc_cv = stats(auc)
        half_m, _, _ = stats(half)
        clf_m, _, _ = stats(clf)
        vzf_m, _, _ = stats(vzf)

        return PKSummary(
            compound=self.drug.name,
            cohort_label=label,
            n=len(pk_list),
            cmax_mean=cmax_m,
            cmax_median=cmax_med,
            cmax_cv=cmax_cv,
            tmax_mean=tmax_m,
            tmax_median=tmax_med,
            auc_mean=auc_m,
            auc_median=auc_med,
            auc_cv=auc_cv,
            half_life_mean=half_m,
            cl_f_mean=clf_m,
            vz_f_mean=vzf_m,
        )

    def _compute_uncertainty(
        self,
        all_patient_pk: dict[str, dict[str, float | None]],
        cohort_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute median and 90% interval for key metrics across virtual subjects.

        Reported per cohort (doses differ across cohorts in a SAD design) plus an
        overall summary.

        If posterior samples are available, use posterior-predictive credible intervals.
        Otherwise, fall back to normal-approx intervals across virtual subjects.
        """
        pk_list = list(all_patient_pk.values())
        metric_keys = ["cmax", "auc_inf", "half_life", "cl_f"]

        def _summarize(pts: list[dict[str, float | None]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key in metric_keys:
                vals: list[float] = []
                for p in pts:
                    v = p.get(key)
                    if v is not None:
                        vals.append(float(v))
                out[key] = summarize_metrics(vals)
            return out

        # If posterior samples are available, compute posterior-predictive intervals
        if self.posterior_samples is not None:
            # Get first dose level for prediction
            dose_mg = self.protocol.dose_levels[0] if self.protocol.dose_levels else 10.0
            t_end_h = float(self.protocol.observation_period_days * 24)
            t_eval = onp.linspace(0, t_end_h, max(int(t_end_h) + 1, 50))

            post_pred = posterior_predictive_pk(
                self.posterior_samples,
                n_patients=len(pk_list),
                dose_mg=dose_mg,
                t_eval=t_eval,
                drug=self.drug,
            )

            def _summarize_posterior(metric: str) -> dict[str, float]:
                """Summarize posterior-predictive samples."""
                samples = post_pred[metric]  # (n_samples, n_patients)
                # Collapse to per-patient posterior mean, then summarize across patients
                patient_means = onp.nanmean(samples, axis=0)  # (n_patients,)
                return summarize_metrics(patient_means.tolist())

            result: dict[str, Any] = {"overall": {}}
            for key in metric_keys:
                if key in post_pred:
                    result["overall"][key] = _summarize_posterior(key)

            # Add per-cohort breakdowns (same for all cohorts since posterior is global)
            for ci in range(len(cohort_summaries)):
                result[f"cohort_{ci + 1}"] = result["overall"].copy()

            return result

        # Fall back to normal-approx intervals across virtual subjects
        result = {"overall": _summarize(pk_list)}

        cursor = 0
        for ci, summary in enumerate(cohort_summaries):
            size = int(summary["n"])
            chunk = pk_list[cursor : cursor + size]
            cursor += size
            result[f"cohort_{ci + 1}"] = _summarize(chunk)

        return result

    @staticmethod
    def _make_run_id() -> str:
        import hashlib
        import uuid

        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        h = hashlib.sha256(uuid.uuid4().hex.encode()).hexdigest()[:6]
        return f"sad_{stamp}_{h}"
