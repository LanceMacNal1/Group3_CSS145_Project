#######################
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import opendatasets as od
import streamlit as st
import builtins

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor


#######################
# Page configuration
st.set_page_config(
    page_title="Customer Personality Analysis", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Customer Personality Analysis')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Marc Dave D. Constantino\n2. Ruskin Gian A. Lauguico\n3. Jean L. Lopez \n4. Lance Nathaniel B. Macalalad \n5. Craig Zyrus B. Manuel")

#######################
# Data

# Load data

dataset_df = pd.read_csv("marketing_campaign.csv", delimiter="\t")
print("Total Rows:", len(dataset_df), "\n")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here
    st.markdown(""" 

    The project will focus on Customer prediction, to be more specific, the most purchased product (Given by the dataset). The group decisded to use K-Means Clustering and Decision Tree to predict our data.

    #### Pages
    1. `Dataset` - This section introduces the Customer Personality dataset, which contains demographic and behavioral information collected from customers. It details the dataset's features and describes how these attributes can help understand customer segmentation and forecast spending habits.
    2. `EDA` - The EDA section analyzes patterns in customer demographics and purchase behaviors, using visualizations like pie charts for product category distributions, bar charts and histograms for spending and demographic trends, a correlation heatmap for feature relationships, and a pairwise scatter plot matrix (if suitable) to explore connections among different attributes.
    3. `Data Cleaning / Pre-processing` - This page outlines the data cleaning and pre-processing procedures, such as handling missing values, encoding categories (e.g., education levels, marital status), and scaling numerical features. The final dataset is then split into training and testing sets to prepare for model training.
    4. `Machine Learning` - In this section, we apply machine learning models to understand customer segments and predict spending. K-Means Clustering is used for customer segmentation, with visualizations to display clusters, while a Random Forest Regressor predicts spending patterns, evaluated using metrics like Mean Absolute Error. Feature importance analysis identifies which factors most significantly affect spending.
    5. `Prediction` - The prediction page provides an interactive tool where users can input data such as customer demographics and preferences to make spending predictions or identify cluster membership using the trained models, offering a real-time application of the analysis.
    6. `Conclusion` - This section highlights key insights from the EDA and model analysis, including observed trends in customer spending and segmentation results. It also suggests future enhancements, like adding granular behavior data or exploring additional models to improve prediction accuracy.


    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.write(dataset_df)

    # Your content for your DATASET page goes here

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    # Filter rows with any null values in any column
    st.write("### Rows with Null Values")
    null_df = dataset_df[dataset_df.isnull().any(axis=1)]
    st.dataframe(null_df)
    
    # Display total rows and null rows
    st.write("### Dataset Summary")
    st.write("Total Rows:", len(dataset_df))
    st.write("Total Null Rows:", len(null_df))
    
    # Drop all rows with null values
    clean_pd = dataset_df.dropna()
    st.write("### Cleaned Dataset")
    st.dataframe(clean_pd)
    
    # Display counts after cleaning
    st.write("Total Not Null Rows:", len(clean_pd))
    
    # Cast columns to proper data types and display
    st.write("### Dataset with Corrected Data Types")
    clean_pd.loc[:, 'ID'] = clean_pd['ID'].astype('int64')
    clean_pd.loc[:, 'Year_Birth'] = pd.to_datetime(clean_pd['Year_Birth'], format='%Y').dt.year
    clean_pd.loc[:, 'Income'] = clean_pd['Income'].astype('float64')
    clean_pd.loc[:, 'Kidhome'] = clean_pd['Kidhome'].astype('int32')
    clean_pd.loc[:, 'Teenhome'] = clean_pd['Teenhome'].astype('int32')
    clean_pd.loc[:, 'Dt_Customer'] = pd.to_datetime(clean_pd['Dt_Customer'], format='%d-%m-%Y').dt.date
    clean_pd.loc[:, 'Recency'] = clean_pd['Recency'].astype('int32')
    clean_pd.loc[:, 'MntWines'] = clean_pd['MntWines'].astype('float64')
    clean_pd.loc[:, 'MntFruits'] = clean_pd['MntFruits'].astype('float64')
    clean_pd.loc[:, 'MntMeatProducts'] = clean_pd['MntMeatProducts'].astype('float64')
    clean_pd.loc[:, 'MntFishProducts'] = clean_pd['MntFishProducts'].astype('float64')
    clean_pd.loc[:, 'MntSweetProducts'] = clean_pd['MntSweetProducts'].astype('float64')
    clean_pd.loc[:, 'MntGoldProds'] = clean_pd['MntGoldProds'].astype('float64')
    clean_pd.loc[:, 'NumDealsPurchases'] = clean_pd['NumDealsPurchases'].astype('int32')
    clean_pd.loc[:, 'NumWebPurchases'] = clean_pd['NumWebPurchases'].astype('int32')
    clean_pd.loc[:, 'NumCatalogPurchases'] = clean_pd['NumCatalogPurchases'].astype('int32')
    clean_pd.loc[:, 'NumStorePurchases'] = clean_pd['NumStorePurchases'].astype('int32')
    clean_pd.loc[:, 'NumWebVisitsMonth'] = clean_pd['NumWebVisitsMonth'].astype('int32')
    
    # Select relevant columns
    clean_pd = clean_pd[['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
                         'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                         'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
    st.dataframe(clean_pd)
    
    # Display data types of columns
    st.write("### Data Types After Casting")
    st.write(clean_pd.dtypes)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
