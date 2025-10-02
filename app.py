"""
Stock Price Prediction App (Streamlit + LSTM)

Features:
- Univariate input (single column prediction)
- Train/Validation/Test split for evaluation
- Shows training history and test predictions
- Provides a fixed 7-day forecast (labeled as Day+1 … Day+7)
- Exports predictions + forecast as CSV
"""

# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from io import BytesIO


# --- Streamlit Page Setup ---
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction with LSTM (7-Day Forecast)")


# --- File Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # --- Load Dataset ---
    df = pd.read_csv(uploaded_file)

    # Clean up columns
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index("Date", inplace=True)

    # --- Dataset Preview ---
    st.subheader("Dataset Preview")
    rows_to_show = st.slider(
        "Select number of rows to preview",
        min_value=5,
        max_value=len(df),
        value=min(20, len(df)),
        step=5
    )
    st.dataframe(df.head(rows_to_show))
    st.caption(f"Showing first {rows_to_show} rows out of {len(df)} total rows.")

    # --- Target Selection (Single Column Only) ---
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for prediction.")
        st.stop()

    target_col = st.selectbox("Select column to predict", numeric_cols)
    data = df[[target_col]].values
    n_features = 1  # univariate

    # --- Hyperparameters ---
    st.sidebar.header("Model Configuration")
    SEQ_LEN = st.sidebar.slider("Sequence Length (days)", 10, 100, 30, step=5)
    EPOCHS = st.sidebar.slider("Training Epochs", 5, 100, 25, step=5)
    BATCH_SIZE = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
    FUTURE_DAYS = 7  # fixed forecast horizon

    # --- Data Preprocessing ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    def create_sequences(dataset, seq_len=SEQ_LEN):
        """Generate sequences for LSTM."""
        X, y = [], []
        for i in range(seq_len, len(dataset)):
            X.append(dataset[i - seq_len:i])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    if len(scaled) <= SEQ_LEN:
        st.error("Dataset too small for selected sequence length.")
        st.stop()

    X_all, y_all = create_sequences(scaled)

    # Train/Validation/Test split: 70/10/20
    train_size = int(len(X_all) * 0.7)
    val_size = int(len(X_all) * 0.1)

    train_X, val_X, test_X = (
        X_all[:train_size],
        X_all[train_size:train_size + val_size],
        X_all[train_size + val_size:]
    )
    train_y, val_y, test_y = (
        y_all[:train_size],
        y_all[train_size:train_size + val_size],
        y_all[train_size + val_size:]
    )

    # --- Build LSTM Model ---
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # --- Train Model ---
    history = model.fit(
        train_X, train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_X, val_y),
        verbose=0
    )

    # --- Training History Plot ---
    st.subheader("Training History")
    fig_hist, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title("Model Training Performance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig_hist)

    # --- Predictions on Test Set ---
    pred_test = model.predict(test_X)
    pred_test = scaler.inverse_transform(pred_test)
    actual_test = scaler.inverse_transform(test_y.reshape(-1, 1))

    # --- Evaluation Metrics ---
    rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
    mape = mean_absolute_percentage_error(actual_test, pred_test) * 100
    mae = mean_absolute_error(actual_test, pred_test)

    st.subheader("Evaluation Metrics")
    st.write(f"- RMSE: **{rmse:.2f}**")
    st.write(f"- MAPE: **{mape:.2f}%**")
    st.write(f"- MAE: **{mae:.2f}**")

    # --- Plot Predictions ---
    st.subheader("Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_test, label="Actual")
    ax.plot(pred_test, label="Predicted")
    ax.set_title(f"{target_col} - Actual vs Predicted")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # --- 7-Day Forecast ---
    st.subheader(f"7-Day Forecast for {target_col}")

    last_seq = scaled[-SEQ_LEN:].copy()
    forecast_scaled = []

    for _ in range(FUTURE_DAYS):
        next_pred = model.predict(last_seq.reshape(1, SEQ_LEN, n_features))[0, 0]
        forecast_scaled.append(next_pred)
        new_entry = np.array([[next_pred]])
        last_seq = np.vstack([last_seq[1:], new_entry])

    forecast_prices = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))

    # --- Forecast Plot ---
    fig_future, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(actual_test)), actual_test, label="Actual (Test)", color="blue")
    ax.plot(range(len(pred_test)), pred_test, label="Predicted (Test)", color="orange")

    # Forecast as Day+N
    forecast_x = [f"Day+{i+1}" for i in range(FUTURE_DAYS)]
    ax.plot(
        range(len(pred_test), len(pred_test) + FUTURE_DAYS),
        forecast_prices,
        "o--",
        color="green",
        label="7-Day Forecast",
        linewidth=2,
        markersize=8
    )
    ax.axvspan(len(pred_test)-1, len(pred_test) + FUTURE_DAYS, color="green", alpha=0.1)

    ax.set_title(f"{target_col} - Test Predictions + 7-Day Forecast", fontsize=14, weight="bold")
    ax.set_xlabel("Timeline")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig_future)

    # --- Forecast Table ---
    forecast_df = pd.DataFrame({
        "Day": forecast_x,
        "Forecasted Price": np.round(forecast_prices.flatten(), 2)
    })

    st.subheader("Forecasted Values")
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # --- Export Results ---
    results_df = pd.DataFrame({
        "Actual": actual_test.flatten(),
        "Predicted": pred_test.flatten()
    })
    combined_df = pd.concat([results_df, forecast_df], axis=1)

    csv_buffer = BytesIO()
    combined_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Predictions + 7-Day Forecast (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"{target_col}_predictions_7day_forecast.csv",
        mime="text/csv"
    )
