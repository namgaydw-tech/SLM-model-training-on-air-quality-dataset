from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "trainmodel"
    / "air_quality_model.pkl"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Trained model not found.")

    print(
        "\nRun train.py first:"
    )

    print(
        "python train.py"
    )

    raise SystemExit(1)


# ============================================================
# LOAD MODEL
# ============================================================

saved_data = joblib.load(
    MODEL_PATH
)

model = saved_data["model"]

features = saved_data["features"]


print()
print("=" * 55)
print("          AIR QUALITY CO PREDICTOR")
print("=" * 55)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_number(message):

    while True:

        value = input(message).strip()

        # Allow European decimal format such as 13,6
        value = value.replace(",", ".")

        try:

            return float(value)

        except ValueError:

            print(
                "Please enter a valid number."
            )


# ============================================================
# DATE
# ============================================================

while True:

    date_input = input(
        "\nEnter date (DD/MM/YYYY): "
    ).strip()

    try:

        date = pd.to_datetime(
            date_input,
            format="%d/%m/%Y",
        )

        break

    except ValueError:

        print(
            "Invalid date. Example: 10/03/2004"
        )


# ============================================================
# TIME
# ============================================================

while True:

    time_input = input(
        "Enter time (HH:MM): "
    ).strip()

    try:

        time_value = pd.to_datetime(
            time_input,
            format="%H:%M",
        )

        hour = time_value.hour

        break

    except ValueError:

        print(
            "Invalid time. Example: 18:00"
        )


# ============================================================
# SENSOR INPUTS
# ============================================================

print(
    "\nEnter the sensor readings:\n"
)


pt08_s1 = get_number(
    "PT08.S1(CO)     : "
)

pt08_s2 = get_number(
    "PT08.S2(NMHC)   : "
)

pt08_s3 = get_number(
    "PT08.S3(NOx)    : "
)

pt08_s4 = get_number(
    "PT08.S4(NO2)    : "
)

pt08_s5 = get_number(
    "PT08.S5(O3)     : "
)

temperature = get_number(
    "Temperature T   : "
)

humidity = get_number(
    "Relative Humidity RH : "
)

absolute_humidity = get_number(
    "Absolute Humidity AH : "
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = {

    "PT08.S1(CO)": pt08_s1,

    "PT08.S2(NMHC)": pt08_s2,

    "PT08.S3(NOx)": pt08_s3,

    "PT08.S4(NO2)": pt08_s4,

    "PT08.S5(O3)": pt08_s5,

    "T": temperature,

    "RH": humidity,

    "AH": absolute_humidity,

    "hour": hour,

    "day_of_week": date.dayofweek,

    "month": date.month,
}


input_df = pd.DataFrame(
    [input_data]
)


# Make sure columns have exactly
# the same order used during training.
input_df = input_df[
    features
]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    input_df
)[0]


print()
print("=" * 55)
print("                 RESULT")
print("=" * 55)

print(
    f"\nPredicted CO(GT): {prediction:.3f}"
)

print()