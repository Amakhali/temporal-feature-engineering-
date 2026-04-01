
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os

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
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df = cyclical_encoding(df, 'hour', 24)
    df = cyclical_encoding(df, 'day_of_week', 7)
    df = cyclical_encoding(df, 'month', 12)

    # Lag features (adjust to use the last actual load value for the first lag)
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

# --- Prediction Function ---
def make_prediction(input_date, hours_to_predict=24):
    # Create a DataFrame for predictions starting from the input_date
    future_dates = pd.date_range(start=input_date, periods=hours_to_predict, freq='H')
    future_df = pd.DataFrame(index=future_dates)

    # Initialize 'load' column for feature creation. We'll fill this with actual/predicted values
    # Use the last known 'load' value from test_data to initialize for the first prediction
    last_known_load_date = test_data.index[-1]
    last_known_load_value = test_data['load'].iloc[-1]

    # Combine test data and future_df to generate features correctly
    # This is crucial for lag and rolling features to propagate correctly
    combined_df = pd.concat([test_data, future_df])

    predictions = []
    for i in range(hours_to_predict):
        current_date = future_df.index[i]

        # For the first prediction, use the last known load value to compute lags/rolling features
        if i == 0 and current_date == input_date:
            temp_df = combined_df.loc[:current_date].copy()
            temp_df.loc[current_date, 'load'] = last_known_load_value # Placeholder, will be replaced by prediction

        else:
            # For subsequent predictions, use previous predicted values for lags
            temp_df = combined_df.loc[:current_date].copy()

            # Fill 'load' with previous predictions for feature generation
            for j, pred_date in enumerate(future_df.index[:i]):
                temp_df.loc[pred_date, 'load'] = predictions[j]

        temp_df = create_features(temp_df.copy())
        current_features = temp_df.loc[current_date].drop('load', errors='ignore')

        # Ensure feature columns match training order and number
        feature_cols = [c for c in test_data.drop('load', axis=1).columns if c in current_features.index]
        current_features = current_features[feature_cols]

        # Handle potential NaNs introduced by feature engineering at the beginning of the prediction horizon
        # For lags, if there isn't enough historical data, we need to decide how to fill.
        # A simple approach is to ffill or use a default, but for robustness, it should be part of training prep.
        # For this demo, we assume enough data for features to be non-NaN after 'test_data'.
        if current_features.isnull().any():
            # If there are NaNs, fill them appropriately. For simplicity here, we'll ffill, but consider more robust imputation.
            current_features = current_features.fillna(method='ffill').fillna(method='bfill') # ffill then bfill for start of series
            # As a fallback, use 0 if still NaN (should not happen with ffill/bfill if any data exists)
            current_features = current_features.fillna(0)

        current_features_scaled = scaler.transform(current_features.values.reshape(1, -1))
        pred = model.predict(current_features_scaled)[0]
        predictions.append(pred)
        combined_df.loc[current_date, 'load'] = pred # Update combined_df with prediction for next iteration's features

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
    value=max_date, # Default to the last date in test data
    min_value=min_date,
    max_value=pd.to_datetime('2025-01-01') # Allow forecasting a bit into the future
)
prediction_start_datetime = pd.to_datetime(prediction_start_date)

hours_to_predict = st.sidebar.slider("Hours to Forecast", 1, 168, 24) # Up to 7 days

if st.sidebar.button("Generate Forecast"):
    st.subheader(f"Energy Consumption Forecast from {prediction_start_datetime.strftime('%Y-%m-%d %H:00')} for {hours_to_predict} hours")

    with st.spinner("Generating forecast..."):
        forecast = make_prediction(prediction_start_datetime, hours_to_predict)

    st.success("Forecast generated!")

    # Display forecast as a table
    st.dataframe(forecast.rename('Forecasted Load').reset_index().rename(columns={'index': 'Datetime'}), use_container_width=True)

    # Plotting the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_data.index[-24*7:], test_data['load'].iloc[-24*7:], label='Recent Actuals', color='blue', alpha=0.7)
    ax.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
    ax.set_title('Energy Consumption Forecast')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Load')
    ax.legend()
    st.pyplot(fig)

    # --- SHAP Explanations ---
    st.subheader("Feature Importance (SHAP Values)")
    st.write("Understanding which features contribute most to the model's predictions.")

    # Use a subset of X_test for SHAP to avoid long computation times
    # Ensure X_test_scaled is available or re-create it
    # For this app, we'll recreate X_test and scale it if not available globally from colab
    # The cached load_resources() ensures test_data is ready.

    # Re-process test_data to get features as done in training
    processed_test_data = create_features(test_data.copy())
    processed_test_data = processed_test_data.dropna() # Drop NaNs from feature creation

    # Align feature columns with training order
    feature_cols = [c for c in processed_test_data.columns if c != 'load']
    X_test_for_shap = processed_test_data[feature_cols]

    if not X_test_for_shap.empty:
        # Select a sample for SHAP for faster calculation
        if len(X_test_for_shap) > 1000:
            X_test_sample = X_test_for_shap.sample(n=1000, random_state=42)
        else:
            X_test_sample = X_test_for_shap

        X_test_sample_scaled = scaler.transform(X_test_sample)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample_scaled)

            st.write("**Summary Plot:**")
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test_sample_scaled, feature_names=X_test_sample.columns, show=False)
            st.pyplot(fig_shap)
            plt.clf() # Clear plot to prevent overlap

            st.write("**Individual Prediction Explanation (First Forecast Point):**")
            # Explain the first point of the generated forecast
            if not forecast.empty:
                first_forecast_date = forecast.index[0]
                # Need to recreate features for this specific point to get the exact scaled input
                # This is a bit complex as make_prediction does it internally. Re-run for just this point.
                temp_df_for_shap = pd.concat([test_data, pd.DataFrame(index=[first_forecast_date])])
                temp_df_for_shap.loc[first_forecast_date, 'load'] = test_data['load'].iloc[-1] # Use last known for initial feature gen
                temp_df_for_shap = create_features(temp_df_for_shap.copy())
                current_features_for_shap = temp_df_for_shap.loc[first_forecast_date].drop('load', errors='ignore')
                current_features_for_shap = current_features_for_shap[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0) # Ensure no NaNs

                current_features_for_shap_scaled = scaler.transform(current_features_for_shap.values.reshape(1, -1))

                # Re-calculate SHAP values for this single instance
                shap_values_single = explainer.shap_values(current_features_for_shap_scaled)

                fig_force, ax_force = plt.subplots(figsize=(10, 2))
                shap.force_plot(explainer.expected_value, shap_values_single[0], current_features_for_shap_scaled[0],
                                feature_names=X_test_sample.columns, matplotlib=True, show=False)
                st.pyplot(fig_force)
                plt.clf()

        except Exception as e:
            st.error(f"Could not generate SHAP plots: {e}. Ensure SHAP is compatible with the model and data structure.")
    else:
        st.warning("Not enough data to generate SHAP explanations.")

else:
    st.info("Select your prediction settings and click 'Generate Forecast'.")
