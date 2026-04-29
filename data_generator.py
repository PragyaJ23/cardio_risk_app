import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic patient data with realistic clinical distributions.
    Labels are derived from a clinically-grounded risk formula.
    """
    rng = np.random.default_rng(random_state)

    age         = rng.integers(20, 85, n_samples).astype(float)
    gender      = rng.integers(0, 2, n_samples)          # 0 = Female, 1 = Male
    bmi         = rng.normal(27, 5, n_samples).clip(16, 50)
    sbp         = rng.normal(125, 20, n_samples).clip(80, 220)
    dbp         = rng.normal(80, 12, n_samples).clip(50, 140)
    heart_rate  = rng.normal(75, 12, n_samples).clip(45, 130)
    med_history = rng.binomial(1, 0.35, n_samples)

    # Inject a small % of missing values to demonstrate preprocessing
    for arr in [age, bmi, sbp, dbp, heart_rate]:
        mask = rng.random(n_samples) < 0.03
        arr[mask] = np.nan

    # Risk score: weighted clinical formula → binary label
    risk_score = (
        0.04  * np.nan_to_num(age)
        + 0.3   * gender
        + 0.05  * np.nan_to_num(bmi)
        + 0.025 * np.nan_to_num(sbp)
        + 0.015 * np.nan_to_num(dbp)
        + 0.01  * np.nan_to_num(heart_rate)
        + 2.5   * med_history
        + rng.normal(0, 0.8, n_samples)
    )
    threshold = np.percentile(risk_score, 60)
    label = (risk_score > threshold).astype(int)

    return pd.DataFrame({
        "Age":         age,
        "Gender":      gender,
        "BMI":         bmi,
        "SBP":         sbp,
        "DBP":         dbp,
        "HeartRate":   heart_rate,
        "MedHistory":  med_history,
        "CardiacRisk": label,
    })
