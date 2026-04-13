import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# Set page config
st.set_page_config(layout="wide", page_title="Energy Consumption Forecast")

# --- Load Model and Data ---
@st.cache_resource
def load_resources():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    test_data = pd.read_csv('test_data.csv', index_col='Datetime', parse_dates=True)
    return model, scaler, test_data

model, scaler, test_data = load_resources()

# --- Feature Engineering (must match training) ---
def cyclical_encoding(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df = cyclical_encoding(df, 'hour', 24)
    df = cyclical_encoding(df, 'day_of_week', 7)
    df = cyclical_encoding(df, 'month', 12)

    # Lag features
    df['lag_1'] = df['load'].shift(1)
    df['lag_2'] = df['load'].shift(2)
    df['lag_3'] = df['load'].shift(3)
    df['lag_24'] = df['load'].shift(24)
    df['lag_168'] = df['load'].shift(168)

    # Rolling statistics (use only past data)
    df['rolling_mean_6h'] = df['load'].shift(1).rolling(6).mean()
    df['rolling_std_6h'] = df['load'].shift(1).rolling(6).std()
    df['rolling_mean_12h'] = df['load'].shift(1).rolling(12).mean()
    df['rolling_std_12h'] = df['load'].shift(1).rolling(12).std()
    df['rolling_mean_24h'] = df['load'].shift(1).rolling(24).mean()
    df['rolling_std_24h'] = df['load'].shift(1).rolling(24).std()

    return df

# Pre‑compute feature columns (exclude 'load')
feature_cols = [c for c in test_data.columns if c != 'load']

# --- Prediction Function (iterative, uses actual + predicted values) ---
def make_prediction(input_date, hours_to_predict=24):
    # Create a DataFrame for predictions
    future_dates = pd.date_range(start=input_date, periods=hours_to_predict, freq='h')
    future_df = pd.DataFrame(index=future_dates)

    # Combine historical data and future dates to have a continuous time series
    # We need the last 168 hours of actual data before input_date to compute initial lags/rolling stats
    history = test_data.loc[test_data.index < input_date].tail(200)  # extra margin
    combined = pd.concat([history, future_df])

    # We will fill the future load values iteratively
    predictions = []

    for i, dt in enumerate(future_dates):
        # Create a window that includes enough history (at least 200 hours back)
        window_start = dt - pd.Timedelta(hours=200)
        window = combined.loc[window_start:dt].copy()

        # Compute features for the window (this fills all columns)
        window = create_features(window)

        # Get the row for the current timestamp (the one we want to predict)
        current_row = window.loc[[dt]].copy()

        # Select only the feature columns (exclude 'load')
        X = current_row[feature_cols]

        # Check for NaNs in features; if any, fill them (should not happen if we have enough history)
        if X.isnull().any().any():
            # Forward fill, then backward fill, then 0
           X = X.ffill().bfill().fillna(0)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]
        predictions.append(pred)

        # Store the predicted load for use in subsequent iterations
        combined.loc[dt, 'load'] = pred

    return pd.Series(predictions, index=future_dates)

# --- Streamlit UI ---
st.title("Hourly Energy Consumption Forecast")
st.write("Forecast future energy consumption based on historical data using an XGBoost model.")

# Sidebar for user inputs
st.sidebar.header("Prediction Settings")

# Date picker for prediction start
max_date = test_data.index.max()
min_date = test_data.index.min()

prediction_start_date = st.sidebar.date_input(
    "Select Prediction Start Date",
    value=max_date,
    min_value=min_date,
    max_value=pd.to_datetime('2025-01-01')
)
prediction_start_datetime = pd.to_datetime(prediction_start_date)

hours_to_predict = st.sidebar.slider("Hours to Forecast", 1, 168, 24)

if st.sidebar.button("Generate Forecast"):
    st.subheader(f"Energy Consumption Forecast from {prediction_start_datetime.strftime('%Y-%m-%d %H:00')} for {hours_to_predict} hours")

    with st.spinner("Generating forecast..."):
        forecast = make_prediction(prediction_start_datetime, hours_to_predict)

    st.success("Forecast generated!")

    # Display forecast as a table - FIXED: use_container_width replaced with width='stretch'
    st.dataframe(forecast.rename('Forecasted Load').reset_index().rename(columns={'index': 'Datetime'}), width='stretch')

    # Plotting the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_data.index[-24*7:], test_data['load'].iloc[-24*7:], label='Recent Actuals', color='blue', alpha=0.7)
    ax.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
    ax.set_title('Energy Consumption Forecast')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Load (MW)')
    ax.legend()
    st.pyplot(fig)

    # --- SHAP Explanations ---
    st.subheader("Feature Importance (SHAP Values)")
    st.write("Understanding which features contribute most to the model's predictions.")

    # Prepare SHAP data (use a sample of the test set)
    processed_test = create_features(test_data.copy()).dropna()
    X_test_shap = processed_test[feature_cols]

    if not X_test_shap.empty:
        # Use a random sample to keep performance reasonable
        sample_size = min(1000, len(X_test_shap))
        X_sample = X_test_shap.sample(n=sample_size, random_state=42)
        X_sample_scaled = scaler.transform(X_sample)

        try:
            # Cache the explainer (expensive)
            @st.cache_resource
            def get_explainer():
                return shap.TreeExplainer(model)

            explainer = get_explainer()
            shap_values = explainer.shap_values(X_sample_scaled)

            st.write("**Summary Plot:**")
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_cols, show=False)
            st.pyplot(fig_shap)
            plt.clf()

            st.write("**Individual Prediction Explanation (First Forecast Point):**")
            # Re‑create the exact input for the first forecast point
            if not forecast.empty:
                first_forecast_date = forecast.index[0]
                # Build a combined dataframe that ends at the first forecast date
                # and includes the last actual value for that timestamp
                temp_df = pd.concat([test_data.tail(200), pd.DataFrame(index=[first_forecast_date])])
                temp_df.loc[first_forecast_date, 'load'] = test_data['load'].iloc[-1]  # initial guess (will be overwritten later)
                temp_df = create_features(temp_df)
                X_first = temp_df.loc[[first_forecast_date], feature_cols]
                # Handle any NaNs (should be rare) - FIXED: replaced fillna(method=...) with ffill/bfill
                X_first = X_first.ffill().bfill().fillna(0)
                X_first_scaled = scaler.transform(X_first)

                shap_first = explainer.shap_values(X_first_scaled)

                # Force plot for this single instance
                fig_force, ax_force = plt.subplots(figsize=(10, 2))
                shap.force_plot(explainer.expected_value, shap_first[0], X_first_scaled[0],
                                feature_names=feature_cols, matplotlib=True, show=False)
                st.pyplot(fig_force)
                plt.clf()
        except Exception as e:
            st.error(f"Could not generate SHAP plots: {e}. Please ensure SHAP is installed and compatible.")
    else:
        st.warning("Not enough data to generate SHAP explanations.")

else:
    st.info("Select your prediction settings and click 'Generate Forecast'.")
