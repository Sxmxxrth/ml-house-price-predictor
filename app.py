import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

@st.cache_resource
def load_data_and_train_model():
    # Load dataset
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['Price'] = california.target
    
    # Train Model
    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return df, model, r2, mse

st.title("🏡 House Price Predictor")
st.markdown("Predict the median house value in California districts using a **Random Forest Regressor**.")

with st.spinner("Loading data and training model..."):
    df, model, r2, mse = load_data_and_train_model()

st.sidebar.header("User Input Features")
st.sidebar.markdown("Use the sliders to adjust the features and predict the price.")

def user_input_features():
    med_inc = st.sidebar.slider("Median Income (in $10k)", float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()))
    house_age = st.sidebar.slider("House Age (years)", float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()))
    ave_rooms = st.sidebar.slider("Average Rooms", float(df['AveRooms'].min()), 10.0, float(df['AveRooms'].mean()))
    ave_bedrms = st.sidebar.slider("Average Bedrooms", float(df['AveBedrms'].min()), 5.0, float(df['AveBedrms'].mean()))
    population = st.sidebar.slider("Population", float(df['Population'].min()), 5000.0, float(df['Population'].mean()))
    ave_occup = st.sidebar.slider("Average Occupancy", float(df['AveOccup'].min()), 10.0, float(df['AveOccup'].mean()))
    latitude = st.sidebar.slider("Latitude", float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].mean()))
    longitude = st.sidebar.slider("Longitude", float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].mean()))
    
    data = {'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Predictive Analytics")
    st.write("Input Features:")
    st.dataframe(input_df)
    
    prediction = model.predict(input_df)
    
    st.success(f"### Predicted Median House Value: ${prediction[0] * 100000:,.2f}")

with col2:
    st.subheader("Model Performance")
    st.info(f"**R² Score:** {r2:.4f}")
    st.info(f"**Mean Squared Error:** {mse:.4f}")

st.markdown("---")
st.subheader("Exploratory Data Analysis (EDA)")

eda_col1, eda_col2 = st.columns(2)
with eda_col1:
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], bins=50, kde=True, ax=ax, color="skyblue")
    ax.set_title("Distribution of House Prices (x $100,000)")
    st.pyplot(fig)

with eda_col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df.sample(1000), x="MedInc", y="Price", alpha=0.5, color="coral")
    ax2.set_title("Median Income vs House Price")
    st.pyplot(fig2)
