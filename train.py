from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

# Folder where train.py exists
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "AirQualityUCI.csv"

MODEL_DIR = BASE_DIR / "trainmodel"
MODEL_PATH = MODEL_DIR / "air_quality_model.pkl"


# ============================================================
# SETTINGS
# ============================================================

TARGET = "CO(GT)"

RANDOM_STATE = 42


# We deliberately use sensor/weather/time information
# instead of other GT pollutant labels.

FEATURES = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "hour",
    "day_of_week",
    "month",
]


# ============================================================
# START
# ============================================================

print()
print("=" * 60)
print("       AIR QUALITY CO PREDICTION MODEL TRAINING")
print("=" * 60)

print("\nDataset path:")
print(DATASET_PATH)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():

    print("\nERROR: Dataset was not found.")
    print("\nExpected location:")
    print(DATASET_PATH)

    print(
        "\nPlace AirQualityUCI.csv inside the data folder."
    )

    raise SystemExit(1)


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    sep=";",
    decimal=",",
)


print(f"Original dataset shape: {df.shape}")


# ============================================================
# REMOVE EMPTY DATA
# ============================================================

# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all",
)

# Remove completely empty rows
df = df.dropna(
    axis=0,
    how="all",
)


print(
    f"Shape after removing empty rows/columns: {df.shape}"
)


# ============================================================
# FIX MISSING VALUES
# ============================================================

# This particular UCI dataset uses -200
# to represent missing sensor measurements.
df = df.replace(
    -200,
    np.nan,
)


# ============================================================
# CREATE DATE/TIME FEATURES
# ============================================================

print("\nCreating time features...")


datetime_text = (
    df["Date"].astype(str)
    + " "
    + df["Time"]
        .astype(str)
        .str.replace(".", ":", regex=False)
)


df["datetime"] = pd.to_datetime(
    datetime_text,
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce",
)


df["hour"] = df["datetime"].dt.hour

df["day_of_week"] = (
    df["datetime"].dt.dayofweek
)

df["month"] = df["datetime"].dt.month


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

df = df.dropna(
    subset=[
        TARGET,
        "datetime",
    ]
)


print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# CREATE X AND Y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


print("\nFeatures being used:")

for feature in FEATURES:
    print(f"  -> {feature}")


print(
    f"\nTarget: {TARGET}"
)


# ============================================================
# RANDOM TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
)


print("\nRandom split:")
print(
    f"Training samples : {len(X_train)}"
)
print(
    f"Testing samples  : {len(X_test)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


model = create_model()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)

print("Training finished.")


# ============================================================
# TEST
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

r2 = r2_score(
    y_test,
    predictions,
)

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)


print()
print("=" * 60)
print("             RANDOM TEST PERFORMANCE")
print("=" * 60)

print(
    f"R² Score : {r2:.4f}"
)

print(
    f"R² %     : {r2 * 100:.2f}%"
)

print(
    f"MAE      : {mae:.4f}"
)

print(
    f"RMSE     : {rmse:.4f}"
)


# ============================================================
# CHRONOLOGICAL TEST
# ============================================================

# Air-quality data is time based.
# This test checks how well the model predicts later readings
# using earlier readings.

print()
print("=" * 60)
print("         CHRONOLOGICAL FUTURE TEST")
print("=" * 60)


time_df = df.sort_values(
    "datetime"
).reset_index(drop=True)


split_index = int(
    len(time_df) * 0.80
)


chrono_train = time_df.iloc[
    :split_index
]

chrono_test = time_df.iloc[
    split_index:
]


chrono_model = create_model()


chrono_model.fit(
    chrono_train[FEATURES],
    chrono_train[TARGET],
)


chrono_predictions = (
    chrono_model.predict(
        chrono_test[FEATURES]
    )
)


chrono_r2 = r2_score(
    chrono_test[TARGET],
    chrono_predictions,
)

chrono_mae = mean_absolute_error(
    chrono_test[TARGET],
    chrono_predictions,
)

chrono_rmse = np.sqrt(
    mean_squared_error(
        chrono_test[TARGET],
        chrono_predictions,
    )
)


print(
    f"R² Score : {chrono_r2:.4f}"
)

print(
    f"R² %     : {chrono_r2 * 100:.2f}%"
)

print(
    f"MAE      : {chrono_mae:.4f}"
)

print(
    f"RMSE     : {chrono_rmse:.4f}"
)


# ============================================================
# TRAIN FINAL MODEL USING ALL DATA
# ============================================================

print()
print("=" * 60)
print("            TRAINING FINAL MODEL")
print("=" * 60)


final_model = create_model()


final_model.fit(
    X,
    y,
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model_package = {

    "model": final_model,

    "features": FEATURES,

    "target": TARGET,

    "random_test_metrics": {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    },

    "chronological_test_metrics": {
        "r2": float(chrono_r2),
        "mae": float(chrono_mae),
        "rmse": float(chrono_rmse),
    },
}


joblib.dump(
    model_package,
    MODEL_PATH,
)


print("\nModel saved successfully!")

print(
    f"\n{MODEL_PATH}"
)


print()
print("=" * 60)
print("                  DONE")
print("=" * 60)