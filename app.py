# app.py - Main application file
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Crypto Price Predictor", layout="wide")

# App title and description
st.title("Cryptocurrency Price Prediction")
st.markdown("""
This application predicts cryptocurrency prices using machine learning.
* Data source: Yahoo Finance
* Features: OHLC, volume, and technical indicators
* Model: Random Forest Regressor
""")

# Sidebar for input parameters
st.sidebar.header('User Input Parameters')
selected_crypto = st.sidebar.selectbox('Cryptocurrency', ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD'])
prediction_days = st.sidebar.slider('Days to Predict', 1, 30, 7)
train_size = st.sidebar.slider('Training Data Size (days)', 30, 365, 180)


# Function to get data
@st.cache_data
def load_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    return data


# Function to create features
def create_features(df):
    # Technical indicators
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['20MA'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
    df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)

    # Volume features
    df['Volume_1d_change'] = df['Volume'].pct_change()
    df['Volume_7d_mean'] = df['Volume'].rolling(window=7).mean()

    # Price features
    df['Price_1d_change'] = df['Close'].pct_change()
    df['Price_7d_change'] = df['Close'].pct_change(periods=7)

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=14).std()

    # Day of week
    df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek

    # Drop NaN values
    df = df.dropna()

    return df


# Function to prepare data for ML model
def prepare_data(df, target_col='Close', prediction_days=7):
    # Create target variable (future price)
    df['Target'] = df[target_col].shift(-prediction_days)

    # Drop rows with NaN in Target
    df = df.dropna()

    # Features and target
    X = df.drop(columns=[col for col in ['Target', 'Open', 'High', 'Low'] if col in df.columns])
    y = df['Target']

    return X, y


# Function to train model
def train_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return model, scaler, X_test, y_test, y_pred, feature_importance


# Function to make future predictions
def predict_future(model, scaler, last_data, days=7):
    future_predictions = []
    last_values = last_data.iloc[-1:].copy()

    for _ in range(days):
        # Scale the data
        scaled_data = scaler.transform(last_values.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1))

        # Make prediction for the next day
        prediction = model.predict(scaled_data)[0]
        future_predictions.append(prediction)

        # Create a new row with the prediction
        new_row = last_values.copy()
        new_row['Close'] = prediction

        # Update features for the new row (this is simplified)
        # In a real application, you would need more sophisticated feature updating
        new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]

        # Add the new row to last_values for the next iteration
        last_values = new_row

    return future_predictions


# Load data
with st.spinner('Fetching cryptocurrency data...'):
    data = load_data(selected_crypto, period='2y')
    st.success(f"Data loaded for {selected_crypto}")

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Process data
with st.spinner('Processing data and creating features...'):
    processed_data = create_features(data)
    st.success("Features created successfully")

# Display processed data
st.subheader('Processed Data with Technical Indicators')
st.write(processed_data.tail())

# Filter data to the training size
processed_data = processed_data.iloc[-train_size:]

# Prepare data for model
X, y = prepare_data(processed_data, prediction_days=prediction_days)

# Train model
with st.spinner('Training model...'):
    model, scaler, X_test, y_test, y_pred, feature_importance = train_model(X, y)
    st.success("Model trained successfully")

# Model evaluation
st.subheader('Model Evaluation')
col1, col2 = st.columns(2)
with col1:
    mae = np.mean(np.abs(y_pred - y_test.values))
    mse = np.mean((y_pred - y_test.values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100

    st.write(f"Mean Absolute Error (MAE): ${mae:.2f}")
    st.write(f"Mean Squared Error (MSE): ${mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

with col2:
    st.subheader('Feature Importance')
    fig = go.Figure(go.Bar(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        orientation='h'
    ))
    fig.update_layout(title='Feature Importance', xaxis_title='Importance', yaxis_title='Feature')
    st.plotly_chart(fig)

# Make future predictions
with st.spinner(f'Predicting prices for the next {prediction_days} days...'):
    future_dates = pd.date_range(start=processed_data.index[-1] + pd.Timedelta(days=1), periods=prediction_days)
    future_predictions = predict_future(model, scaler, processed_data, days=prediction_days)

    # Create DataFrame for predictions
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions
    }).set_index('Date')

    st.success(f"Predictions generated for the next {prediction_days} days")

# Display predictions
st.subheader('Price Predictions')
st.write(prediction_df)

# Plot historical prices and predictions
st.subheader('Historical Prices and Predictions')
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Historical prices
fig.add_trace(
    go.Scatter(
        x=processed_data.index,
        y=processed_data['Close'],
        name='Historical Price',
        line=dict(color='blue')
    )
)

# Predictions
fig.add_trace(
    go.Scatter(
        x=prediction_df.index,
        y=prediction_df['Predicted_Price'],
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    )
)

# Layout
fig.update_layout(
    title=f'{selected_crypto} Price Prediction',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Technical indicators visualization
st.subheader('Technical Indicators')
tech_indicator = st.selectbox('Select Technical Indicator',
                              ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands'])

# Plot based on selection
if tech_indicator == 'Moving Averages':
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA7'], name='MA7'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA14'], name='MA14'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA30'], name='MA30'))

elif tech_indicator == 'RSI':
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Price', 'RSI'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['RSI'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)

elif tech_indicator == 'MACD':
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Price', 'MACD'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], name='Close'), row=1, col=1)
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Signal'], name='Signal'), row=2, col=1)

else:  # Bollinger Bands
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close'], name='Close'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Upper_Band'], name='Upper Band'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['20MA'], name='20-day MA'))
    fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Lower_Band'], name='Lower Band'))

fig.update_layout(title=f'{tech_indicator} for {selected_crypto}', xaxis_title='Date')
st.plotly_chart(fig, use_container_width=True)

# Add download button for the prediction data
csv = prediction_df.to_csv()
st.download_button(
    label="Download Prediction Data as CSV",
    data=csv,
    file_name=f'{selected_crypto}_prediction_{dt.datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)

# Model download option
if st.button('Download Trained Model'):
    # Save model to file
    model_filename = f'{selected_crypto}_model.joblib'
    joblib.dump(model, model_filename)

    # Provide download link
    with open(model_filename, 'rb') as file:
        st.download_button(
            label='Download Model File',
            data=file,
            file_name=model_filename,
            mime='application/octet-stream'
        )

# Instructions for running the application
st.subheader('How to Run This Application')
st.code('''
# Install required packages
pip install streamlit pandas numpy scikit-learn plotly yfinance joblib

# Save the code above as app.py
# Run the application
streamlit run app.py
''')

# About section
st.sidebar.markdown('---')
st.sidebar.subheader('About')
st.sidebar.info('''
This app demonstrates:
- Data collection from Yahoo Finance
- Feature engineering with technical indicators
- Machine learning model training
- Interactive visualization
- Cryptocurrency price prediction
''')