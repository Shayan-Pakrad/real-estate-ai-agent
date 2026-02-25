import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title('Data Analysis and Model Prediction')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    # Data Analysis Section
    if st.checkbox('Show Dataframe Summary'):
        st.write(df.describe())

    if st.checkbox('Show Correlation Heatmap'):
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()

    # Prepare data
    st.subheader('Train/Test Split and Model Selection')
    target_col = st.selectbox('Select the target column', df.columns)
    features = df.drop(columns=[target_col])

    X = features
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Model Evaluation
    st.subheader('Model Evaluation')
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    # Visualize Predictions vs Actual
    st.subheader('Prediction vs Actual')
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    st.pyplot()
