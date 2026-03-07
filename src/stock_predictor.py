import yfinance as yf
import os
import plotly.express as px
import plotly.graph_objects as go

from neuralprophet import NeuralProphet

# Download 1 year of Apple stock data
df = yf.download("AAPL", period="5y")

print("\\nFull DataFrame:\\n")
print(df.head())

df = df.reset_index()[["Date", "Close"]]
df.columns = ["ds", "y"]

print(f"\\nTotal dataset size: {len(df)} rows")

train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

print(f"\\nTraining set size: {len(train_df)} rows")
print(f"Test set size: {len(test_df)} rows")

m = NeuralProphet(learning_rate=0.01)

print("\\nStarting training...\\n")
metrics = m.fit(train_df, freq="D")

print("\\n\\nPredicting on test set...\\n")
forecast = m.predict(test_df)

print("\\nPrediction complete!")
print(f"Forecast shape: {forecast.shape}")
print(f"\\nFirst 5 predictions:\\n{forecast[['ds', 'y', 'yhat1']].head()}")

os.makedirs("plots", exist_ok=True)

print("\\n\\nCreating plots...")

forecast_train = m.predict(train_df)
forecast_train = forecast_train[["ds", "y", "yhat1"]]

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="lines",
        name="Training Data",
        line=dict(color="blue", width=1),
    )
)
fig1.add_trace(
    go.Scatter(
        x=forecast_train["ds"],
        y=forecast_train["yhat1"],
        mode="lines",
        name="Training Predictions",
        line=dict(color="cyan", dash="dash", width=1),
    )
)
fig1.add_trace(
    go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="lines",
        name="Test Data",
        line=dict(color="orange", width=1),
    )
)
fig1.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat1"],
        mode="lines",
        name="Test Predictions",
        line=dict(color="red", dash="dash", width=1.5),
    )
)
fig1.update_layout(
    title="AAPL Stock Price: Training and Test Predictions",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
)
fig1.write_html("plots/stock_predictions.html")
print("Saved: plots/stock_predictions.html")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="lines",
        name="Actual",
        line=dict(color="orange", width=1),
    )
)
fig2.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat1"],
        mode="lines",
        name="Predicted",
        line=dict(color="red", dash="dash", width=1.5),
    )
)
fig2.update_layout(
    title="AAPL Stock Price: Test Set Actual vs Predicted",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
)
fig2.write_html("plots/test_predictions.html")
print("Saved: plots/test_predictions.html")

forecast_plot = forecast.copy()
forecast_plot["residuals"] = forecast_plot["y"] - forecast_plot["yhat1"]

fig3 = px.scatter(
    forecast_plot,
    x="ds",
    y="residuals",
    title="Prediction Residuals",
    labels={"ds": "Date", "residuals": "Residual (Actual - Predicted)"},
)
fig3.update_traces(marker=dict(size=4, color="purple", opacity=0.7))
fig3.add_hline(y=0, line_dash="dash", line_color="gray")
fig3.update_layout(template="plotly_white", height=400)
fig3.write_html("plots/residuals.html")
print("Saved: plots/residuals.html")

fig4 = px.histogram(
    forecast_plot,
    x="residuals",
    nbins=50,
    title="Distribution of Residuals",
    labels={"residuals": "Residual (Actual - Predicted)"},
)
fig4.update_layout(template="plotly_white", height=400)
fig4.write_html("plots/residuals_distribution.html")
print("Saved: plots/residuals_distribution.html")

print("\\nAll plots saved to 'plots' folder!")
