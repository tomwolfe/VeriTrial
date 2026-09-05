"""Core Pydantic v2 schemas for the InSilico Clinical Trial Simulator.

All parameters in the simulator must flow through these schemas. No magic numbers
should exist outside config files that load into these models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SexEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class DosingRoute(str, Enum):
    ORAL = "oral"
    IV = "iv"
    SUBCUTANEOUS = "subcutaneous"
    INHALATION = "inhalation"


class TrialDesign(str, Enum):
    SAD = "SAD"
    MAD = "MAD"
    BATCH = "batch"


class GenotypeEnum(str, Enum):
    INTERMEDIATE = "intermediate"
    POOR = "poor"
    EXTENSIVE = "extensive"
    ULTRA_RAPID = "ultra_rapid"


# ---------------------------------------------------------------------------
# Drug Schema
# ---------------------------------------------------------------------------


class Drug(BaseModel):
    """Pharmacological properties of a drug candidate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mol_weight: float = Field(..., description="Molecular weight (g/mol)", gt=0)
    log_p: float = Field(..., description="Octanol-water partition coefficient")
    pka: list[float] = Field(default_factory=list, description="pKa values (any ionizable groups)")
    fup: float = Field(..., description="Fraction unbound in plasma", ge=0.0, le=1.0)
    bp_ratio: float = Field(..., description="Blood-to-plasma partition ratio", gt=0)
    dose_unit: str = Field(default="mg")

    # PK parameters (population-typical). Units convention (documented in docs/ASSUMPTIONS.md):
    #   typical_cl_f : total apparent clearance CL/F (L/h) for a 70 kg reference adult
    #   typical_v_f  : total apparent volume Vz/F (L) for a 70 kg reference adult
    # Patient values are scaled allometrically by weight and (for CL) by the
    # metabolizer activity score of ``metabolizing_enzyme``.
    typical_cl_f: float = Field(..., description="Total apparent clearance CL/F (L/h), 70 kg reference adult", gt=0)
    typical_v_f: float = Field(..., description="Total apparent volume Vz/F (L), 70 kg reference adult", gt=0)
    ka: float = Field(..., description="Absorption rate constant (1/h)", gt=0)
    bioavailability: float = Field(..., description="Absolute oral bioavailability", ge=0.0, le=1.0)
    metabolizing_enzyme: str = Field(default="cyp3a4", description="Gene whose activity score scales metabolic clearance (e.g. cyp2c9, cyp3a4; 'none' disables)")

    # PD parameters
    target: str = Field(default="unnamed_target")
    ec50: float = Field(..., description="Half-maximal effective concentration", gt=0)
    emax: float = Field(..., description="Maximum effect", gt=0)
    hill_coeff: float = Field(default=1.0, description="Hill coefficient", gt=0)

    # Safety parameters
    qtcd_baseline: float = Field(default=400.0, description="Baseline QTc interval (ms)")
    qtcd_emax: float = Field(default=0.0, description="Max QTc prolongation from drug effect (ms)")
    qtcd_ec50: float = Field(default=0.0, description="Plasma conc for half-max QTc effect (same units as ec50)")
    qtcd_slope: float = Field(default=0.0, description="Linear QTc-concentration slope (ms per conc unit)")
    dili_risk: float = Field(default=0.01, description="Baseline DILI risk probability", ge=0.0, le=1.0)

    # DILI exposure-response: drives simulated ALT/bilirubin from liver exposure
    alt_baseline: float = Field(default=22.0, description="Baseline ALT (U/L)")
    bili_baseline: float = Field(default=0.8, description="Baseline total bilirubin (mg/dL)")
    dili_emax_alt: float = Field(default=1.5, description="Max ALT multiple over baseline", ge=0.0)
    dili_ec50_alt: float = Field(default=8.0, description="Liver exposure (AUC, mg*h/L) for half-max ALT")
    dili_emax_bili: float = Field(default=1.2, description="Max bilirubin multiple over baseline", ge=0.0)
    dili_ec50_bili: float = Field(default=12.0, description="Liver exposure (AUC, mg*h/L) for half-max bilirubin")

    # QSP DILI parameters (optional; when absent, assess_dili uses empirical proxy)
    vmax_metabolic: float = Field(default=0.0, description="Max metabolic rate for QSP DILI model (mg/h)", ge=0.0)
    km_metabolic: float = Field(default=0.0, description="Michaelis constant for metabolic activation (mg/L)", ge=0.0)
    gsh_depletion_rate: float = Field(default=0.0, description="GSH depletion rate constant for QSP model (L/mg/h)", ge=0.0)

    @property
    def has_qsp_dili_params(self) -> bool:
        """True if QSP DILI parameters are set (non-zero)."""
        return self.vmax_metabolic > 0.0 or self.km_metabolic > 0.0 or self.gsh_depletion_rate > 0.0

    @field_validator("pka", mode="before")
    @classmethod
    def validate_pka(cls, v: Any) -> list[float]:
        if not isinstance(v, list):
            return []
        return v


# ---------------------------------------------------------------------------
# Patient Schema
# ---------------------------------------------------------------------------


class ActivityScore(BaseModel):
    """CYP genotype activity score."""

    gene: str
    allele: str
    activity_score: float = Field(..., ge=0.0, le=2.0)
    metabolizer_status: GenotypeEnum


class Biometric(BaseModel):
    """Anthropometric and physiological measurements."""

    age: float = Field(..., ge=18.0, le=120.0)
    sex: SexEnum
    weight: float = Field(..., gt=0.0)  # kg
    height: float = Field(..., gt=0.0)  # cm
    egfr: float = Field(..., gt=0.0)    # mL/min/1.73m2

    @property
    def bmi(self) -> float:
        return self.weight / ((self.height / 100.0) ** 2)

    @property
    def bsa(self) -> float:
        """Body surface area (Du Bois formula)."""
        return float(0.007184 * (self.weight ** 0.425) * (self.height ** 0.725))


class Patient(BaseModel):
    """A single virtual patient with demographics, biometrics, and genotype."""

    model_config = ConfigDict(extra="forbid")

    id: str
    biometrics: Biometric
    genotypes: dict[str, ActivityScore]
    # Derived scaling factors
    weight_scaling: float = Field(default=1.0, description="Allometric scaling factor")
    age_scaling: float = Field(default=1.0, description="Age-based scaling factor")
    egfr_scaling: float = Field(default=1.0, description="Renal function scaling factor")

    @property
    def bmi(self) -> float:
        return self.biometrics.bmi

    @property
    def bsa(self) -> float:
        return self.biometrics.bsa

    @property
    def is_male(self) -> bool:
        return self.biometrics.sex == SexEnum.MALE


class Population(BaseModel):
    """A virtual population of patients."""

    model_config = ConfigDict(extra="forbid")

    name: str
    n_subjects: int = Field(..., ge=1)
    patients: list[Patient] = Field(..., min_length=1)
    config_hash: str = Field(default="", description="Hash of the generating config")

    def __len__(self) -> int:
        return len(self.patients)


# ---------------------------------------------------------------------------
# Protocol Schema
# ---------------------------------------------------------------------------


class VisitSpec(BaseModel):
    """A scheduled clinical visit."""

    model_config = ConfigDict(extra="forbid")

    day: int
    time: float  # hours from first dose (0 = pre-dose)
    description: str


class DosingEvent(BaseModel):
    """A single dosing event in a MAD protocol."""

    model_config = ConfigDict(extra="forbid")
    time_h: float = Field(..., ge=0, description="Hours from trial start")
    dose_mg: float = Field(..., gt=0)
    route: DosingRoute = DosingRoute.ORAL


class DoseEscalationRule(BaseModel):
    """Rules for dose escalation in SAD/MAD trials."""

    model_config = ConfigDict(extra="forbid")

    rule: str = Field(default="modified_accrual", description="Escalation rule name")
    max_dlt_per_cohort: int = Field(default=1, ge=0)
    min_dlt_free_days: int = Field(default=7, ge=0)
    next_dose_multiplier: float = Field(default=2.0, gt=1.0)
    starting_dose: float = Field(..., gt=0)
    max_dose: float | None = None
    max_administered_doses: int | None = None


class DropoutSpec(BaseModel):
    """Dropout specification."""

    model_config = ConfigDict(extra="forbid")

    rate_per_day: float = Field(default=0.0, ge=0.0, le=1.0)
    cause: str = Field(default="protocol", description="Primary cause of dropout")


class AdherenceSpec(BaseModel):
    """Adherence specification."""

    model_config = ConfigDict(extra="forbid")

    distribution: str = Field(default="uniform")
    min: float = Field(default=0.85, ge=0.0, le=1.0)
    max: float = Field(default=1.0, ge=0.0, le=1.0)


class MeasurementNoiseSpec(BaseModel):
    """Measurement noise specification."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["lognormal", "normal", "additive"] = "lognormal"
    cv_percent: float | None = None  # for lognormal: coefficient of variation
    sd_percent: float | None = None  # for normal/additive: standard deviation as % of value
    sd_absolute: float | None = None  # absolute SD


class SafetyThresholds(BaseModel):
    """Safety monitoring thresholds."""

    model_config = ConfigDict(extra="forbid")

    qt_threshold: float = Field(default=500.0, description="Absolute QTc threshold (ms)")
    qt_delta: float = Field(default=60.0, description="Delta QTc threshold (ms)")
    alt_threshold: float = Field(default=3.0, description="ALT threshold (x ULN)")
    bilirubin_threshold: float = Field(default=2.0, description="Bilirubin threshold (x ULN)")
    ctcae_version: float = Field(default=5.0)


class Protocol(BaseModel):
    """A clinical trial protocol specification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: str = Field(default="Phase I")
    design: TrialDesign = TrialDesign.SAD
    n_cohorts: int = Field(default=1, ge=1)
    cohort_size: int = Field(default=10, ge=1)
    dose_levels: list[float] = Field(..., min_length=1)
    dose_unit: str = Field(default="mg")
    dosing_route: DosingRoute = DosingRoute.ORAL
    dose_escalation: DoseEscalationRule
    dosing_interval_days: int = Field(default=1, ge=1)
    observation_period_days: float = Field(default=7.0, gt=0)
    visit_schedule: list[VisitSpec]
    dropout: DropoutSpec
    adherence: AdherenceSpec
    measurement_noise: MeasurementNoiseSpec
    safety: SafetyThresholds
    solver: str = Field(default="diffrax", description="PBPK ODE solver: 'diffrax' or 'fixed_step'")
    dosing_events: list[DosingEvent] = Field(
        default_factory=list,
        description="Explicit dosing events for MAD; if empty, auto-generated from dose_levels + dosing_interval_days"
    )
    n_doses: int | None = Field(default=None, description="Number of doses for MAD (auto-generates dosing_events if provided)")

    @property
    def doses_mg(self) -> list[float]:
        return self.dose_levels


class Observation(BaseModel):
    """A single measurement from a patient at a time point."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str
    time: float  # hours from dose
    compartment: str
    concentration: float | None = None  # ng/mL or mg/L
    qt_interval: float | None = None     # ms (for ECG)
    pr_interval: float | None = None     # ms
    alt: float | None = None             # U/L
    bilirubin: float | None = None       # mg/dL
    adverse_event: str | None = None
    ctcae_grade: int | None = None       # 0-5
    dlt: bool = False                    # dose-limiting toxicity
    notes: str | None = None


class PopulationSummary(BaseModel):
    """Summary statistics for a population."""

    model_config = ConfigDict(extra="forbid")

    n: int
    mean_age: float
    std_age: float
    n_male: int
    n_female: int
    mean_weight: float
    std_weight: float
    mean_bmi: float
    median_egfr: float
    mean_cl: float
    std_cl: float
    mean_v: float
    std_v: float
    mean_cmax: float | None = None


class PKSummary(BaseModel):
    """PK summary statistics (NCA-like)."""

    model_config = ConfigDict(extra="forbid")

    compound: str
    cohort_label: str
    n: int
    cmax_mean: float | None = None
    cmax_median: float | None = None
    cmax_cv: float | None = None
    tmax_mean: float | None = None
    tmax_median: float | None = None
    auc_mean: float | None = None
    auc_median: float | None = None
    auc_cv: float | None = None
    half_life_mean: float | None = None
    cl_f_mean: float | None = None
    vz_f_mean: float | None = None
    pct_with_dlt: float | None = None


class TrialResult(BaseModel):
    """Top-level result container for a trial simulation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    protocol_name: str
    drug_name: str
    population_name: str
    n_subjects: int
    n_cohorts: int
    pk_summaries: list[PKSummary]
    population_summary: PopulationSummary | None = None
    safety_summary: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    cohort_summaries: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config Loading Utilities
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    p = Path(path)
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {p} did not produce a dict at top level")
    return data


def load_drug_config(path: str | Path) -> Drug:
    """Load drug config from YAML into a validated Drug schema."""
    raw = load_yaml(path)
    return Drug.model_validate(raw["drug"])


def load_population_config(path: str | Path) -> dict[str, Any]:
    """Load population config from YAML."""
    raw = load_yaml(path)
    return dict(raw["population"])


def load_protocol_config(path: str | Path) -> Protocol:
    """Load protocol config from YAML into a validated Protocol schema."""
    raw = load_yaml(path)
    return Protocol.model_validate(raw["protocol"])


def load_observation_json(path: str | Path) -> list[Observation]:
    """Load observations from a JSONL or JSON file."""
    p = Path(path)
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [Observation.model_validate(d) for d in data]


__all__ = [
    # Enums
    "SexEnum",
    "DosingRoute",
    "TrialDesign",
    "GenotypeEnum",
    # Schemas
    "Drug",
    "ActivityScore",
    "Biometric",
    "Patient",
    "Population",
    "VisitSpec",
    "DosingEvent",
    "DoseEscalationRule",
    "DropoutSpec",
    "AdherenceSpec",
    "MeasurementNoiseSpec",
    "SafetyThresholds",
    "Protocol",
    "Observation",
    "PopulationSummary",
    "PKSummary",
    "TrialResult",
    # Loaders
    "load_yaml",
    "load_drug_config",
    "load_population_config",
    "load_protocol_config",
    "load_observation_json",
]
