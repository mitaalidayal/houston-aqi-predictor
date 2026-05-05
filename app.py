import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ============================================================
# LOAD FILES
# ============================================================

MODEL_PATH = "rf_final_aqi.pkl"
SCHEMA_PATH = "feature_schema.json"
DATA_PATH = "recent_aqi_data.csv"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features_for_prediction(recent_data):
    df = recent_data.copy().sort_values("date").reset_index(drop=True)

    df["AQI_lag_1"] = df["AQI"].shift(1)
    df["AQI_lag_2"] = df["AQI"].shift(2)
    df["AQI_lag_7"] = df["AQI"].shift(7)

    df["AQI_diff_1"] = df["AQI"] - df["AQI"].shift(1)
    df["AQI_diff_2"] = df["AQI"] - df["AQI"].shift(2)
    df["AQI_diff_7"] = df["AQI"] - df["AQI"].shift(7)

    df["AQI_rising_1"] = (df["AQI_diff_1"] > 0).astype(int)
    df["AQI_rising_2"] = (df["AQI_diff_2"] > 0).astype(int)
    df["AQI_rising_7"] = (df["AQI_diff_7"] > 0).astype(int)

    df["AQI_momentum_3"] = (
        (df["AQI"].shift(1) > df["AQI"].shift(2)) &
        (df["AQI"].shift(2) > df["AQI"].shift(3))
    ).astype(int)

    df["AQI_ma_3"] = df["AQI"].shift(1).rolling(3).mean()
    df["AQI_ma_7"] = df["AQI"].shift(1).rolling(7).mean()
    df["AQI_std_7"] = df["AQI"].shift(1).rolling(7).std()

    df["temp_x_lowwind"] = df["temperature"] * (1 / (df["wind_speed"] + 1))
    df["temperature_lag_1"] = df["temperature"].shift(1)
    df["humidity_lag_1"] = df["humidity"].shift(1)
    df["wind_speed_lag_1"] = df["wind_speed"].shift(1)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["defining_parameter_lag_1"] = df["defining_parameter"].shift(1)

    for poll in ["NO2", "Ozone", "PM10", "PM2.5"]:
        col = f"poll_yday_{poll}"
        df[col] = (df["defining_parameter_lag_1"] == poll).astype(int)

    return df


def predict_next_day_aqi(recent_data):
    feat_df = build_features_for_prediction(recent_data)
    last_row = feat_df.iloc[[-1]]

    X_pred = last_row[schema["feature_columns"]]
    proba = model.predict_proba(X_pred)[0]

    threshold = schema["threshold"]
    pred_class = int(np.argmax(proba))

    if proba[2] >= threshold:
        pred_class = 2

    return pred_class, proba


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Houston AQI Predictor", layout="wide")

st.title("Houston Next-Day AQI Predictor")

st.write(
    "Enter today's AQI and weather conditions to estimate tomorrow's air quality category for Houston."
)

st.caption(
    "Note: The model was trained on 2019–2024 Houston AQI and weather data. "
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Enter Today's Conditions")

latest = df.iloc[-1]

today_date = pd.Timestamp(
    st.sidebar.date_input(
        "Today's Date",
        value=pd.to_datetime("2026-04-01").date()
    )
)

today_aqi = st.sidebar.slider(
    "Today's AQI",
    min_value=0,
    max_value=250,
    value=int(latest["AQI"])
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=-5.0,
    max_value=45.0,
    value=float(latest["temperature"])
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=int(latest["humidity"])
)

wind_speed_mph = st.sidebar.slider(
    "Wind Speed (mph)",
    min_value=0.0,
    max_value=40.0,
    value=round(float(latest["wind_speed"]) * 0.621371, 1)
)

# Convert mph to km/h because the model was trained on km/h
wind_speed = wind_speed_mph / 0.621371

precipitation = st.sidebar.slider(
    "Precipitation (mm)",
    min_value=0.0,
    max_value=80.0,
    value=float(latest["precipitation"])
)

# Users usually do not know the defining pollutant.
# PM2.5 is used as the default because it is the most common pollutant in the training data.
pollutant = "PM2.5"

# ============================================================
# CREATE CUSTOM RECENT WINDOW
# ============================================================

history = df.tail(29).copy()

custom_today = pd.DataFrame([{
    "date": today_date,
    "AQI": today_aqi,
    "defining_parameter": pollutant,
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "precipitation": precipitation
}])

recent_window = pd.concat([history, custom_today], ignore_index=True)

pred_class, proba = predict_next_day_aqi(recent_window)

label = schema["class_names"][pred_class]
description = schema["class_descriptions"][str(pred_class)]
prediction_date = today_date + pd.Timedelta(days=1)

# ============================================================
# OUTPUT
# ============================================================

st.subheader("Tomorrow's AQI Prediction")

col1, col2, col3 = st.columns(3)

col1.metric("Predicted Category", label)
col2.metric("Confidence", f"{max(proba) * 100:.1f}%")
col3.metric("Prediction Date", prediction_date.date().isoformat())

st.write(description)

if label == "USG+":
    st.warning(
        "Air quality may be unhealthy for sensitive groups. Children, older adults, "
        "and people with respiratory conditions should reduce prolonged outdoor activity."
    )
elif label == "Moderate":
    st.info(
        "Air quality is expected to be acceptable for most people, though unusually sensitive individuals may want to monitor conditions."
    )
else:
    st.success("Air quality is expected to be good.")

# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

st.subheader("Prediction Probability Distribution")

prob_df = pd.DataFrame({
    "Category": schema["class_names"],
    "Probability": proba
})

st.bar_chart(prob_df.set_index("Category"))

st.caption(f"USG+ threshold used: {schema['threshold']}")

# ============================================================
# TREND CHART WITH CUSTOM INPUT
# ============================================================

st.subheader("Recent AQI Trend With Your Input")

trend_df = recent_window.tail(30)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(trend_df["date"], trend_df["AQI"], marker="o", linewidth=1)
ax.axhline(50, linestyle="--", alpha=0.6, label="Good / Moderate boundary")
ax.axhline(100, linestyle="--", alpha=0.6, label="USG+ boundary")

ax.scatter(
    custom_today["date"],
    custom_today["AQI"],
    s=100,
    label="Your input"
)

ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.set_title("Recent AQI Pattern Used for Prediction")
ax.legend()

st.pyplot(fig)

# ============================================================
# INPUT SUMMARY
# ============================================================

st.subheader("Your Input Summary")

display_input = custom_today.copy()
display_input["wind_speed_mph"] = wind_speed_mph
display_input = display_input.drop(columns=["wind_speed"])

st.dataframe(display_input, use_container_width=True)

# ============================================================
# MODEL DETAILS
# ============================================================

st.subheader("Model Details")

st.write("Model: Final Random Forest")
st.write("Prediction target: Next-day AQI category")
st.write("Main pollutant assumption: PM2.5")
st.write(f"Features used: {schema['n_features']}")
st.write(f"Training range: {schema['training_date_range'][0]} to {schema['training_date_range'][1]}")
st.write(f"USG+ threshold: {schema['threshold']}")