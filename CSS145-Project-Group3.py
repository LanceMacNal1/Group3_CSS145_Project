import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import opendatasets as od

# Page Configuration
st.set_page_config(
    page_title="Customer Personality Analysis Dashboard", 
    page_icon="assets/icon.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar Navigation
with st.sidebar:
    st.title('Customer Personality Analysis')
    
    st.subheader("Pages")
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"
    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"
    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"
    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"
    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"
    
    st.subheader("Members")
    st.markdown("1. Elon Musk\n2. Jeff Bezos\n3. Sam Altman\n4. Mark Zuckerberg")

########################
# Functions

# Load and process the dataset
@st.cache_data
def load_data():
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
    
    return y_test, y_pred, X_test, model

# Display features and feature importance
def display_feature_importance(model, X, y):
    feature_importances = {}
    overall_feature_importances = {feature: 0 for feature in X.columns}

    for i, target in enumerate(y.columns):
        importances = model.estimators_[i].feature_importances_
        for feature, importance in zip(X.columns, importances):
            overall_feature_importances[feature] += importance

    num_targets = len(y.columns)
    for feature in overall_feature_importances:
        overall_feature_importances[feature] /= num_targets
    
    st.write("Feature Importances by Product:")
    feature_importance_df = pd.DataFrame.from_dict(overall_feature_importances, orient="index", columns=["Importance"])
    st.bar_chart(feature_importance_df.sort_values(by="Importance", ascending=False))

########################
# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown("""
    **Project Overview:**
    This project aims to analyze customer personality and predict their purchasing behavior using machine learning techniques.
    
    **Dataset:**
    The dataset used is the Customer Personality Analysis from Kaggle, which contains information about customers' demographics and their purchasing behaviors.
    
    **Team Members:**
    1. Elon Musk
    2. Jeff Bezos
    3. Sam Altman
    4. Mark Zuckerberg
    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.write("Here is a preview of the dataset:")
    dataset_df = load_data()
    st.dataframe(dataset_df.head())

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    col = st.columns((1.5, 4.5, 2), gap='medium')
    
    # Example plots (you can customize with your own)
    with col[0]:
        st.markdown('#### Distribution of Income')
        sns.histplot(dataset_df['Income'], kde=True)
        st.pyplot()

    with col[1]:
        st.markdown('#### Correlation Heatmap')
        corr_matrix = dataset_df.corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot()

    with col[2]:
        st.markdown('#### Education vs. Marital Status')
        sns.countplot(x='Education', hue='Marital_Status', data=dataset_df)
        st.pyplot()

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Pre-processing")
    dataset_df = load_data()
    clean_pd = preprocess_data(dataset_df)
    st.write("Cleaned Data Preview:")
    st.dataframe(clean_pd.head())

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Train model
    dataset_df = load_data()
    clean_pd = preprocess_data(dataset_df)
    y_test, y_pred, X_test, model = train_predict_model(clean_pd)

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

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("🔮 Prediction")
    st.write("In this section, you can make individual predictions based on new customer data.")

# Conclusion Page
elif st.session_state.page_selection == "conclusion":
    st.header("🔚 Conclusion")
    st.markdown("""
    This dashboard offers a comprehensive overview of customer personality analysis and product purchasing behavior prediction using machine learning.
    The RandomForest model demonstrated satisfactory performance with the current dataset, with further improvements possible through hyperparameter tuning.
    """)

