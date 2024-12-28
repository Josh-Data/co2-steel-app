import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(
    page_title="Steel Industry CO2 Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv("Steel_industry_data.csv")
        
        # Convert date column to datetime and extract features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df = df.drop('date', axis=1)
            
        return df, None
    except Exception as e:
        return None, str(e)

def preprocess_data(df, is_training=False):
    """Handle categorical variables and ensure proper data types"""
    df = df.copy()
    
    # Drop Day_of_week if present
    if 'Day_of_week' in df.columns:
        df = df.drop(columns=['Day_of_week'])
    
    # Handle categorical variables
    if 'WeekStatus' in df.columns:
        df['WeekStatus'] = df['WeekStatus'].str.strip()
        weekstatus_dummies = pd.get_dummies(df['WeekStatus'], prefix='WeekStatus')
        # Ensure all expected columns are present
        for col in ['WeekStatus_Weekday', 'WeekStatus_Weekend']:
            if col not in weekstatus_dummies.columns:
                weekstatus_dummies[col] = 0
        df = pd.concat([df, weekstatus_dummies], axis=1)
        df = df.drop('WeekStatus', axis=1)
    
    if 'Load_Type' in df.columns:
        df['Load_Type'] = df['Load_Type'].str.strip().str.replace(' ', '_')
        loadtype_dummies = pd.get_dummies(df['Load_Type'], prefix='Load_Type')
        # Ensure all expected columns are present
        for col in ['Load_Type_Light_Load', 'Load_Type_Medium_Load', 'Load_Type_Maximum_Load']:
            if col not in loadtype_dummies.columns:
                loadtype_dummies[col] = 0
        df = pd.concat([df, loadtype_dummies], axis=1)
        df = df.drop('Load_Type', axis=1)
    
    # Ensure all numeric columns are float64
    numeric_columns = [
        'Usage_kWh', 
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh', 
        'Lagging_Current_Power_Factor',
        'Leading_Current_Power_Factor',
        'NSM',
        'year',
        'month',
        'day',
        'hour'
    ]
    
    for col in df.columns:
        if col != 'CO2(tCO2)' and (col in numeric_columns or col.startswith(('WeekStatus_', 'Load_Type_'))):
            df[col] = df[col].astype(np.float64)
    
    return df

def train_model(df):
    """Train the XGBoost model"""
    co2_column = 'CO2(tCO2)'
    co2_values = df[co2_column].astype(np.float64)
    
    # Process data
    df_processed = preprocess_data(df, is_training=True)
    df_processed[co2_column] = co2_values
    
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    X = trainer.drop(columns=[co2_column])
    y = trainer[co2_column]
    
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42, shuffle=False)
    
    # Change to XGBRegressor instead of Classifier
    model = xgb.XGBRegressor(
        n_estimators=25,
        learning_rate=0.1,
        max_depth=7,
        subsample=1.0
    )
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model, tester, X.columns

# [Previous plotting functions remain the same]

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    df, error = load_data()
    if error:
        st.error(f"Error loading data: {error}")
        return
        
    if df is None:
        st.error("Could not load the dataset!")
        return
        
    st.success("Data loaded successfully!")
    
    # Training section
    st.header("Model Training")
    if st.button("Train Model"):
        with st.spinner("Training in progress... 🔄"):
            try:
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                st.success("Model trained successfully! 🎉")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                return

    # Prediction Interface
    if 'model' in st.session_state:
        st.header("Predict CO2 Emissions")
        
        # Time-based inputs
        st.subheader("Date Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.selectbox("Year", options=[2018, 2019], index=0)
            month = st.selectbox("Month", options=list(range(1, 13)), index=0)
            day = st.selectbox("Day", options=list(range(1, 32)), index=0)
        
        with col2:
            hour = st.selectbox("Hour", options=list(range(24)), index=12)
            nsm = hour * 3600  # Convert hour to seconds since midnight
        
        with col3:
            week_status = st.selectbox("Week Status", options=["Weekday", "Weekend"])
            load_type = st.selectbox("Load Type", options=["Light_Load", "Medium_Load", "Maximum_Load"])
        
        # [Rest of the UI and prediction code remains the same]

if __name__ == "__main__":
    main()