import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Load and process the dataset
@st.cache
def load_data():
    import opendatasets as od
    od.download("https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
    dataset_df = pd.read_csv("customer-personality-analysis/marketing_campaign.csv", delimiter="\t")
    return dataset_df

# Data Cleaning and Preprocessing
def preprocess_data(dataset_df):
    clean_pd = dataset_df.dropna()
    clean_pd.loc[:, 'ID'] = clean_pd['ID'].astype('int64')
    clean_pd['Year_Birth'] = pd.to_datetime(clean_pd['Year_Birth'], format='%Y').dt.year
    clean_pd['Income'] = clean_pd['Income'].astype('float64')
    clean_pd['Kidhome'] = clean_pd['Kidhome'].astype('int32')
    clean_pd['Teenhome'] = clean_pd['Teenhome'].astype('int32')
    clean_pd['Recency'] = clean_pd['Recency'].astype('int32')
    
    # Selecting relevant columns for analysis
    clean_pd = clean_pd[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                         'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                         'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
    return clean_pd

# Feature engineering for KMeans
def encode_data_for_kmeans(clean_pd):
    label_encoders = {}
    columns_to_encode = ['Year_Birth', 'Marital_Status', 'Education']
    for col in columns_to_encode:
        le = LabelEncoder()
        clean_pd[col] = le.fit_transform(clean_pd[col])
        label_encoders[col] = le
    return clean_pd, label_encoders

# Machine learning model (RandomForestRegressor)
def train_predict_model(clean_pd):
    classify_pd = clean_pd[['Year_Birth', 'Marital_Status', 'Kidhome', 'Teenhome', 'Education', 'Income', 'Recency',
                            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
    classify_pd = pd.get_dummies(classify_pd, columns=['Marital_Status', 'Education'], drop_first=True)

    X = classify_pd.drop(['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], axis=1)
    y = classify_pd[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, y_pred, X_test

# Display features and feature importance
def display_feature_importance(model, X, y):
    feature_importances = {}
    individual_feature_importance_list = []
    overall_feature_importances = {feature: 0 for feature in X.columns}

    for i, target in enumerate(y.columns):
        importances = model.estimators_[i].feature_importances_
        feature_importances[target] = dict(zip(X.columns, importances))
        for feature, importance in zip(X.columns, importances):
            overall_feature_importances[feature] += importance

    num_targets = len(y.columns)
    for feature in overall_feature_importances:
        overall_feature_importances[feature] /= num_targets
    
    st.write("Feature Importances by Product:")
    feature_importance_df = pd.DataFrame.from_dict(overall_feature_importances, orient="index", columns=["Importance"])
    st.bar_chart(feature_importance_df.sort_values(by="Importance", ascending=False))

# Main Streamlit interface
def main():
    st.title("Customer Personality Analysis and Prediction")
    
    # Load and display data
    dataset_df = load_data()
    st.write("Dataset Preview:")
    st.dataframe(dataset_df.head())
    
    # Data Preprocessing
    clean_pd = preprocess_data(dataset_df)
    st.write("Cleaned Data Preview:")
    st.dataframe(clean_pd.head())

    # Train model
    y_test, y_pred, X_test = train_predict_model(clean_pd)

    # Display Model Performance
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error: {mae}")

    # Display feature importance
    display_feature_importance(model, X_test, y_test)
    
    # Display predictions
    st.write("Predicted vs Actual Values:")
    results_df = pd.DataFrame({
        'Actual': y_test.mean(),
        'Predicted': y_pred.mean(axis=0)
    })
    st.dataframe(results_df)
    
    # Visualization (Example: Correlation heatmap)
    st.write("Correlation Heatmap between Income and Product Spending")
    corr_matrix = clean_pd[['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot()

if __name__ == "__main__":
    main()
