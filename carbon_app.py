import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from datetime import datetime, time
import os

# Set page config for dark theme
st.set_page_config(
    page_title="Steel Industry CO2 Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #111111;
        color: #00ffcc;
    }
    </style>
    """, unsafe_allow_html=True)

# Debug information at startup
st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv("Steel_industry_data.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
            df.set_index('date', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

# Feature order definition
FEATURE_ORDER = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
    'year',
    'month',
    'day',
    'hour',
    'dayofweek',
    'WeekStatus_Weekday',
    'WeekStatus_Weekend',
    'Load_Type_Light_Load',
    'Load_Type_Medium_Load',
    'Load_Type_Maximum_Load'
]

def preprocess_data(df, is_training=False):
    """
    Handle categorical variables and add time features with strict feature ordering
    """
    df = df.copy()
    
    # Drop Day_of_week if present
    if 'Day_of_week' in df.columns:
        df = df.drop(columns=['Day_of_week'])
    
    # Clean categorical values
    if 'Load_Type' in df.columns:
        df['Load_Type'] = df['Load_Type'].str.replace(' ', '_')
    
    # Create zero-filled dummy columns first
    dummy_cols = {
        'WeekStatus_Weekday': 0,
        'WeekStatus_Weekend': 0,
        'Load_Type_Light_Load': 0,
        'Load_Type_Medium_Load': 0,
        'Load_Type_Maximum_Load': 0
    }
    
    for col in dummy_cols:
        df[col] = 0
    
    # Fill in the appropriate dummy values
    if 'WeekStatus' in df.columns:
        for idx, row in df.iterrows():
            df.at[idx, f'WeekStatus_{row["WeekStatus"]}'] = 1
    
    if 'Load_Type' in df.columns:
        for idx, row in df.iterrows():
            df.at[idx, f'Load_Type_{row["Load_Type"]}'] = 1
    
    # Drop original categorical columns
    df = df.drop(columns=['WeekStatus', 'Load_Type'], errors='ignore')
    
    # Ensure all numeric columns are float64
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)
    
    # Reorder columns to match expected feature order
    available_features = [col for col in FEATURE_ORDER if col in df.columns]
    df = df[available_features]
    
    return df

def train_model(df):
    """Train the XGBoost model"""
    df_processed = preprocess_data(df, is_training=True)
    
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    X = trainer.drop(columns=["CO2(tCO2)"])
    y = trainer["CO2(tCO2)"].astype(np.float64)
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42, shuffle=False)
    
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

def plot_predictions(tester, model):
    """Plot actual vs predicted values"""
    X_test = tester.drop(columns=["CO2(tCO2)"])
    predictions = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester["CO2(tCO2)"],
        name="Actual",
        line=dict(color="#00ffcc", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=predictions,
        name="Predicted",
        line=dict(color="#ff00ff", width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="CO2 Emissions: Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="CO2 Emissions (tCO2)",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#00ffcc"),
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 204, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 204, 0.2)'
        )
    )
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=features_df['Importance'],
        y=features_df['Feature'],
        orientation='h',
        marker=dict(color="#00ffcc")
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#00ffcc"),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 204, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 255, 204, 0.2)'
        )
    )
    return fig

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    # Load data with error handling
    df, error = load_data()
    if error:
        st.error(f"Error loading data: {error}")
        st.write("Please make sure 'Steel_industry_data.csv' is in the correct location.")
        return
        
    if df is None:
        st.error("Could not load the dataset.")
        return
        
    st.success("Data loaded successfully!")
    st.write("Dataset Shape:", df.shape)
    st.write("First few rows of the dataset:")
    st.write(df.head())
    
    # Training section
    st.header("Model Training")
    if st.button("Train Model"):
        with st.spinner("Training in progress... 🔄"):
            model, test_data, feature_names = train_model(df)
            st.session_state['model'] = model
            st.session_state['test_data'] = test_data
            st.session_state['feature_names'] = feature_names
            st.success("Model trained successfully! 🎉")
    
    # Rest of your code (prediction interface, etc.) remains the same...
    # I'm truncating it here for brevity, but keep all the prediction interface code
    
if __name__ == "__main__":
    main()