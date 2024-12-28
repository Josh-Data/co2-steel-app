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

def add_logo():
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            background-image: url(logo.png);
            background-repeat: no-repeat;
            padding-top: 120px;
            background-position: 20px 20px;
        }
        .stApp {
            background-color: #fafcff;
        }
        
        /* Base styles for all text */
        .stMarkdown, .stText, .stSelectbox label, .stSlider label, .st-emotion-cache-1vbkxwb e1f1d6gn0 {
            color: #2c3e50 !important;
        }
        
        /* Slider styles */
       .st-emotion-cache-1y4p8pa {
           background-color: #4addbe !important;
       }
       
       /* Slider track before thumb */
       .st-emotion-cache-1y4p8pa > div > div > div > div[style*="background"] {
           background-color: #4addbe !important;
       }
       
       /* Slider thumb */
       .st-emotion-cache-1y4p8pa > div > div > div > div > div[role="slider"] {
           background-color: #4addbe !important;
       }
       
       /* Slider track after thumb */
       .st-emotion-cache-1y4p8pa > div > div > div {
           background-color: #e5e5e5 !important;
       }
        
        /* Button styles */
        button[kind="primary"] {
            background-color: #4addbe !important;
            color: white !important;
        }
        
        /* Select box and dropdown styles */
        .stSelectbox > div > div {
            background-color: white !important;
            color: #2c3e50 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50 !important;
        }
        
        /* Success/error messages */
        .stSuccess, .stError {
            color: #2c3e50 !important;
        }
        
        /* All selectbox options */
        div[role="listbox"] span {
            color: #2c3e50 !important;
        }
        
        /* Ensure dropdown text is visible */
        .stSelectbox div[role="button"] {
            color: #2c3e50 !important;
        }
        
        /* Style for numbers/values displayed */
        .st-emotion-cache-1vbkxwb {
            color: #2c3e50 !important;
        }
        
        /* Ensure all input labels are visible */
        label.st-emotion-cache-1whb5pu {
            color: #2c3e50 !important;
        }
        
        /* Style for widget labels */
        .st-emotion-cache-10trblm {
            color: #2c3e50 !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

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
    
    # Define expected column orders
    expected_weekstatus = ['WeekStatus_Weekday', 'WeekStatus_Weekend']
    expected_loadtype = ['Load_Type_Light_Load', 'Load_Type_Medium_Load', 'Load_Type_Maximum_Load']
    
    # Handle categorical variables
    if 'WeekStatus' in df.columns:
        df['WeekStatus'] = df['WeekStatus'].str.strip()
        weekstatus_dummies = pd.get_dummies(df['WeekStatus'], prefix='WeekStatus')
        # Ensure all expected columns are present and in correct order
        for col in expected_weekstatus:
            if col not in weekstatus_dummies.columns:
                weekstatus_dummies[col] = 0
        weekstatus_dummies = weekstatus_dummies[expected_weekstatus]
        df = pd.concat([df, weekstatus_dummies], axis=1)
        df = df.drop('WeekStatus', axis=1)
    
    if 'Load_Type' in df.columns:
        df['Load_Type'] = df['Load_Type'].str.strip().str.replace(' ', '_')
        loadtype_dummies = pd.get_dummies(df['Load_Type'], prefix='Load_Type')
        # Ensure all expected columns are present and in correct order
        for col in expected_loadtype:
            if col not in loadtype_dummies.columns:
                loadtype_dummies[col] = 0
        loadtype_dummies = loadtype_dummies[expected_loadtype]
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
    
    # Ensure consistent column order
    expected_columns = [
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
        'WeekStatus_Weekday',
        'WeekStatus_Weekend',
        'Load_Type_Light_Load',
        'Load_Type_Medium_Load',
        'Load_Type_Maximum_Load'
    ]
    
    # Only include columns that exist in the dataframe
    columns_to_use = [col for col in expected_columns if col in df.columns]
    if 'CO2(tCO2)' in df.columns:
        columns_to_use.append('CO2(tCO2)')
    
    return df[columns_to_use]

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

def plot_predictions(tester, model):
    """Plot actual vs predicted values"""
    co2_column = 'CO2(tCO2)'
    if co2_column not in tester.columns:
        st.error(f"CO2 column '{co2_column}' not found for plotting!")
        return None
    
    X_test = tester.drop(columns=[co2_column])
    predictions = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=tester[co2_column],
        name="Actual",
        line=dict(color="#4addbe", width=2)
    ))
    fig.add_trace(go.Scatter(
        y=predictions,
        name="Predicted",
        line=dict(color="#2c3e50", width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="CO2 Emissions: Actual vs Predicted",
        xaxis_title="Sample",
        yaxis_title="CO2 Emissions (tCO2)",
        plot_bgcolor="#fafcff",
        paper_bgcolor="#fafcff",
        font=dict(color="#2c3e50"),
        showlegend=True
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
        marker=dict(color="#4addbe")
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor="#fafcff",
        paper_bgcolor="#fafcff",
        font=dict(color="#2c3e50"),
        xaxis=dict(
            title=dict(font=dict(color="#34495e")),  # Darker charcoal for x-axis title
            tickfont=dict(color="#34495e")          # Darker charcoal for x-axis ticks
        ),
        yaxis=dict(
            title=dict(font=dict(color="#34495e")),  # Darker charcoal for y-axis title
            tickfont=dict(color="#34495e")          # Darker charcoal for y-axis ticks
        )
    )
    return fig

def main():
    st.title("Steel Industry CO2 Emissions Predictor")
    
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
        with st.spinner("Training in progress..."):
            try:
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                st.success("Model trained successfully!")
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
        
        # Power usage features
        st.subheader("Power Usage Features")
        col1, col2 = st.columns(2)
        
        with col1:
            usage_kwh = st.slider("Usage (kWh)", 
                                min_value=float(df['Usage_kWh'].min()),
                                max_value=float(df['Usage_kWh'].max()),
                                value=float(df['Usage_kWh'].mean()))
                                
            lagging_current = st.slider("Lagging Current Reactive Power",
                                      min_value=float(df['Lagging_Current_Reactive.Power_kVarh'].min()),
                                      max_value=float(df['Lagging_Current_Reactive.Power_kVarh'].max()),
                                      value=float(df['Lagging_Current_Reactive.Power_kVarh'].mean()))
            
            lagging_pf = st.slider("Lagging Current Power Factor",
                                 min_value=float(df['Lagging_Current_Power_Factor'].min()),
                                 max_value=float(df['Lagging_Current_Power_Factor'].max()),
                                 value=float(df['Lagging_Current_Power_Factor'].mean()))
        
        with col2:
            leading_current = st.slider("Leading Current Reactive Power",
                                      min_value=float(df['Leading_Current_Reactive_Power_kVarh'].min()),
                                      max_value=float(df['Leading_Current_Reactive_Power_kVarh'].max()),
                                      value=float(df['Leading_Current_Reactive_Power_kVarh'].mean()))
            
            leading_pf = st.slider("Leading Current Power Factor",
                                 min_value=float(df['Leading_Current_Power_Factor'].min()),
                                 max_value=float(df['Leading_Current_Power_Factor'].max()),
                                 value=float(df['Leading_Current_Power_Factor'].mean()))
        
        if st.button("Predict CO2 Emissions"):
            try:
                input_data = {
                    'Usage_kWh': usage_kwh,
                    'Lagging_Current_Reactive.Power_kVarh': lagging_current,
                    'Leading_Current_Reactive_Power_kVarh': leading_current,
                    'Lagging_Current_Power_Factor': lagging_pf,
                    'Leading_Current_Power_Factor': leading_pf,
                    'NSM': nsm,
                    'year': year,
                    'month': month,
                    'day': day,
                    'hour': hour,
                    'WeekStatus': week_status,
                    'Load_Type': load_type
                }
                
                input_df = pd.DataFrame([input_data])
                input_processed = preprocess_data(input_df, is_training=False)
                
                prediction = st.session_state['model'].predict(input_processed)[0]
                st.success(f"Predicted CO2 Emissions: {prediction:.2f} tCO2")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Debug - Input Features:", input_processed.columns.tolist())
        
        # Show visualizations
        st.header("Model Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predictions vs Actual Values")
            pred_fig = plot_predictions(
                st.session_state['test_data'],
                st.session_state['model']
            )
            if pred_fig:
                st.plotly_chart(pred_fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Importance")
            imp_fig = plot_feature_importance(
                st.session_state['model'],
                st.session_state['feature_names']
            )
            st.plotly_chart(imp_fig, use_container_width=True)

if __name__ == "__main__":
    main()