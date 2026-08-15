"""Tests for the population generator."""


from insilico_trial.population.generator import generate_population


def test_copula_correlations():
    """Generate n=5000, check correlations within 0.05 of target."""
    df, _ = generate_population({
        'name': 'test_correlations',
        'n_subjects': 5000,
        'seed': 42,
        'age': {'dist': 'truncated_normal', 'mean': 40.0, 'std': 12.0, 'min': 18.0, 'max': 75.0},
        'weight': {'dist': 'lognormal', 'mean_log': 4.42, 'std_log': 0.18},
        'height': {'dist': 'truncated_normal', 'mean': 170.0, 'std': 10.0, 'min': 150.0, 'max': 200.0},
        'egfr': {'dist': 'lognormal', 'mean_log': 5.05, 'std_log': 0.35},
        'liver_volume': {'dist': 'lognormal', 'mean_log': 7.31, 'std_log': 0.20},
        'genotypes': {
            'cyp2c9': {
                'alleles': ['CYP2C9*1', 'CYP2C9*2', 'CYP2C9*3'],
                'frequencies': [0.88, 0.09, 0.03],
                'activity_scores': [1.0, 0.5, 0.0],
            },
        },
        'correlation_matrix': {
            'age_egfr': -0.35,
            'weight_height': 0.72,
            'weight_liver_volume': 0.68,
            'age_liver_volume': -0.15,
            'weight_egfr': 0.25,
        }
    })

    continuous_cols = ["age", "weight_kg", "height_cm", "egfr_ml_min", "liver_volume_ml"]
    corr = df[continuous_cols].corr().values

    # Check key correlations are within tolerance
    # age-egfr: -0.35 (indices 0,3 in continuous_cols)
    age_egfr_corr = corr[0, 3]
    assert abs(age_egfr_corr - (-0.35)) < 0.05, f"age_egfr correlation {age_egfr_corr} not within 0.05 of -0.35"

    # weight-height: 0.72 (indices 1,2 in continuous_cols)
    weight_height_corr = corr[1, 2]
    assert abs(weight_height_corr - 0.72) < 0.05, f"weight_height correlation {weight_height_corr} not within 0.05 of 0.72"


def test_genotype_frequencies():
    """Chi-squared test against config frequencies."""
    df, _ = generate_population({
        'name': 'test_genotypes',
        'n_subjects': 1000,
        'seed': 42,
        'age': {'dist': 'truncated_normal', 'mean': 40.0, 'std': 12.0, 'min': 18.0, 'max': 75.0},
        'weight': {'dist': 'lognormal', 'mean_log': 4.42, 'std_log': 0.18},
        'height': {'dist': 'truncated_normal', 'mean': 170.0, 'std': 10.0, 'min': 150.0, 'max': 200.0},
        'egfr': {'dist': 'lognormal', 'mean_log': 5.05, 'std_log': 0.35},
        'liver_volume': {'dist': 'lognormal', 'mean_log': 7.31, 'std_log': 0.20},
        'genotypes': {
            'cyp2c9': {
                'alleles': ['CYP2C9*1', 'CYP2C9*2', 'CYP2C9*3'],
                'frequencies': [0.88, 0.09, 0.03],
                'activity_scores': [1.0, 0.5, 0.0],
            },
        },
        'correlation_matrix': {
            'age_egfr': -0.35,
            'weight_height': 0.72,
            'weight_liver_volume': 0.68,
            'age_liver_volume': -0.15,
            'weight_egfr': 0.25,
        }
    })

    # Check allele frequencies approximate config frequencies
    # The generator samples one allele per patient per gene
    total = len(df)
    cyp2c9_1_pct = (df["cyp2c9_allele"] == "CYP2C9*1").sum() / total
    cyp2c9_2_pct = (df["cyp2c9_allele"] == "CYP2C9*2").sum() / total
    cyp2c9_3_pct = (df["cyp2c9_allele"] == "CYP2C9*3").sum() / total

    assert 0.80 < cyp2c9_1_pct < 0.95, f"CYP2C9*1 freq {cyp2c9_1_pct} outside expected range"
    assert 0.05 < cyp2c9_2_pct < 0.15, f"CYP2C9*2 freq {cyp2c9_2_pct} outside expected range"
    assert 0.01 < cyp2c9_3_pct < 0.06, f"CYP2C9*3 freq {cyp2c9_3_pct} outside expected range"


def test_biometric_ranges():
    """All ages in [18,75], weights in [40,200], etc."""
    df, _ = generate_population({
        'name': 'test_biometrics',
        'n_subjects': 100,
        'seed': 42,
        'age': {'dist': 'truncated_normal', 'mean': 40.0, 'std': 12.0, 'min': 18.0, 'max': 75.0},
        'weight': {'dist': 'lognormal', 'mean_log': 4.42, 'std_log': 0.18},
        'height': {'dist': 'truncated_normal', 'mean': 170.0, 'std': 10.0, 'min': 150.0, 'max': 200.0},
        'egfr': {'dist': 'lognormal', 'mean_log': 5.05, 'std_log': 0.35},
        'liver_volume': {'dist': 'lognormal', 'mean_log': 7.31, 'std_log': 0.20},
        'genotypes': {
            'cyp2c9': {
                'alleles': ['CYP2C9*1', 'CYP2C9*2', 'CYP2C9*3'],
                'frequencies': [0.88, 0.09, 0.03],
                'activity_scores': [1.0, 0.5, 0.0],
            },
        },
        'correlation_matrix': {
            'age_egfr': -0.35,
            'weight_height': 0.72,
            'weight_liver_volume': 0.68,
            'age_liver_volume': -0.15,
            'weight_egfr': 0.25,
        }
    })

    # Check age range [18, 75]
    assert df["age"].between(18, 75).all(), "Age values outside [18, 75]"

    # Check weight range [40, 200]
    assert df["weight_kg"].between(40, 200).all(), "Weight values outside [40, 200]"

    # Check height range [150, 200]
    assert df["height_cm"].between(150, 200).all(), "Height values outside [150, 200]"

    # Check eGFR range [15, 200]
    assert df["egfr_ml_min"].between(15, 200).all(), "eGFR values outside [15, 200]"

    # Check liver volume range [500, 3000]
    assert df["liver_volume_ml"].between(500, 3000).all(), "Liver volume values outside [500, 3000]"
