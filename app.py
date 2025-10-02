"""
Stock Price Prediction App (Streamlit + LSTM)

Features:
- Supports univariate & multivariate input
- Train/Validation/Test split for evaluation
- Shows training history and test predictions
- Provides a fixed 7-day forecast (continuous with test data)
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
uploaded_file = st.file_uploader("📂 Upload your stock dataset (CSV format)", type="csv")

if uploaded_file:

    # --- Load Dataset ---
    df = pd.read_csv(uploaded_file)

    # Clean up column names
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --- Feature & Target Selection ---
    feature_cols = st.multiselect(
        "Select input features (X)",
        df.columns[1:],
        default=[df.columns[1]]
    )
    target_col = st.selectbox("Select target column (y)", df.columns[1:])

    # Always include target in features
    if target_col not in feature_cols:
        feature_cols.append(target_col)

    if not feature_cols:
        st.error("⚠️ Please select at least one feature column.")
        st.stop()

    data = df[feature_cols].values
    n_features = len(feature_cols)
    target_index = feature_cols.index(target_col)

    # --- Hyperparameters (sidebar) ---
    st.sidebar.header("⚙️ Model Configuration")
    SEQ_LEN = st.sidebar.slider("Sequence Length (days)", 10, 100, 30, step=5)
    EPOCHS = st.sidebar.slider("Training Epochs", 5, 100, 25, step=5)
    BATCH_SIZE = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
    FUTURE_DAYS = 7   # fixed forecast horizon

    # --- Data Preprocessing ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    def create_sequences(dataset, seq_len=SEQ_LEN):
        """Generate input/output sequences for LSTM."""
        X, y = [], []
        for i in range(seq_len, len(dataset)):
            X.append(dataset[i - seq_len:i])
            y.append(dataset[i, target_index])
        return np.array(X), np.array(y)

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

    # --- Plot Training History ---
    st.subheader("📉 Training History")
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

    def invert_scaling(scaled_vals, scaler, feature_index):
        """Convert scaled values back to original price scale."""
        scaled_vals = np.array(scaled_vals).reshape(-1, 1)
        filler = np.zeros((scaled_vals.shape[0], scaler.scale_.shape[0]))
        filler[:, feature_index] = scaled_vals.flatten()
        inv = scaler.inverse_transform(filler)
        return inv[:, feature_index]

    pred_test = invert_scaling(pred_test, scaler, target_index)
    actual_test = invert_scaling(test_y, scaler, target_index)

    # --- Evaluation Metrics ---
    rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
    mape = mean_absolute_percentage_error(actual_test, pred_test) * 100
    mae = mean_absolute_error(actual_test, pred_test)

    st.subheader("📊 Evaluation Metrics")
    st.write(f"- **RMSE:** {rmse:.2f}")
    st.write(f"- **MAPE:** {mape:.2f}%")
    st.write(f"- **MAE:** {mae:.2f}")

    # --- Plot Predictions ---
    st.subheader("📈 Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_test, label="Actual")
    ax.plot(pred_test, label="Predicted")
    ax.set_title(f"{target_col} - Actual vs Predicted")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # --- 7-Day Forecast ---
    st.subheader(f"🔮 7-Day Forecast for {target_col}")

    last_seq = scaled[-SEQ_LEN:].copy()
    forecast_scaled = []

    for _ in range(FUTURE_DAYS):
        next_pred = model.predict(last_seq.reshape(1, SEQ_LEN, n_features))[0, 0]
        forecast_scaled.append(next_pred)

        # Update rolling window with predicted value
        new_entry = np.zeros((1, n_features))
        new_entry[0, target_index] = next_pred
        last_seq = np.vstack([last_seq[1:], new_entry])

    forecast_prices = invert_scaling(forecast_scaled, scaler, target_index)

    # Plot Forecast continuous with last test values
    fig_future, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(actual_test)), actual_test, label="Actual (Test)")
    ax.plot(range(len(pred_test)), pred_test, label="Predicted (Test)")
    ax.plot(
        range(len(pred_test), len(pred_test) + FUTURE_DAYS),
        forecast_prices,
        marker="o",
        label="7-Day Forecast"
    )
    ax.set_title(f"{target_col} - Test Predictions + 7-Day Forecast")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig_future)

    st.success(f"📈 Forecasted 7-Day Prices: {np.round(forecast_prices, 2)}")

    # --- Export Results ---
    results_df = pd.DataFrame({
        "Actual": actual_test.flatten(),
        "Predicted": pred_test.flatten()
    })
    forecast_df = pd.DataFrame({
        "Day": [f"Day+{i+1}" for i in range(FUTURE_DAYS)],
        "Forecast": forecast_prices.flatten()
    })

    combined_df = pd.concat([results_df, forecast_df], axis=1)

    csv_buffer = BytesIO()
    combined_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Predictions + 7-Day Forecast (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"{target_col}_predictions_7day_forecast.csv",
        mime="text/csv"
    )
