# 🏡 House Price Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)

An interactive Machine Learning web application built with **Streamlit** and **Scikit-Learn** that predicts California housing prices based on user inputs.

## 🚀 Features
- **Live Interactive UI**: Adjust features like Median Income, House Age, and Population using sliders.
- **Machine Learning Model**: Powered by a robust `RandomForestRegressor` for high accuracy.
- **Real-time EDA**: View data distributions and correlations dynamically within the dashboard.

## 🛠️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sxmxxrth/ml-house-price-predictor.git
   cd ml-house-price-predictor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 🧠 Machine Learning Details
The model uses the California Housing Dataset. The data is split 80/20 for training and testing. We achieve an R² score of ~0.80 using an ensemble learning technique (Random Forest) which significantly outperforms standard Linear Regression by capturing non-linear relationships.

## 📝 License
MIT License