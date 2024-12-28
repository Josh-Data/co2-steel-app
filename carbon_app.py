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

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv("Steel_industry_data.csv")
        st.write("Original columns in dataset:", df.columns.tolist())
        return df, None
    except Exception as e:
        return None, str(e)

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
    
    return df

def train_model(df):
    """Train the XGBoost model"""
    st.write("Training data columns:", df.columns.tolist())
    
    # Find CO2 column with exact name
    co2_column = 'CO2(tCO2)'  # Use exact column name from the dataset
    if co2_column not in df.columns:
        st.error(f"CO2 column '{co2_column}' not found! Available columns:")
        st.write(df.columns.tolist())
        raise ValueError(f"CO2 column '{co2_column}' not found in dataset")
    
    st.write(f"Using {co2_column} as target variable")
    
    # Store CO2 values before preprocessing
    co2_values = df[co2_column].copy()
    
    # Process data
    df_processed = preprocess_data(df, is_training=True)
    st.write("Processed data columns:", df_processed.columns.tolist())
    
    # Add CO2 values back
    df_processed[co2_column] = co2_values
    
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    X = trainer.drop(columns=[co2_column])
    y = trainer[co2_column].astype(np.float64)
    
    st.write("Feature columns for training:", X.columns.tolist())
    
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
        line=dict(color="#00ffcc", width=2)
    ))
    fig.add_trace(go.Scatter(
        y=predictions,
        name="Predicted",
        line=dict(color="#ff00ff", width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="CO2 Emissions: Actual vs Predicted",
        xaxis_title="Sample",
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
            
            nsm = st.slider("NSM (Seconds since midnight)", 
                          min_value=0,
                          max_value=86400,
                          value=43200)
        
        # Categorical features
        st.subheader("Categorical Features")
        col1, col2 = st.columns(2)
        
        with col1:
            week_status = st.selectbox("Week Status", options=["Weekday", "Weekend"])
        
        with col2:
            load_type = st.selectbox("Load Type", options=["Light_Load", "Medium_Load", "Maximum_Load"])
        
        if st.button("Predict CO2 Emissions"):
            try:
                # Create prediction input
                input_data = {
                    'Usage_kWh': usage_kwh,
                    'Lagging_Current_Reactive.Power_kVarh': lagging_current,
                    'Leading_Current_Reactive_Power_kVarh': leading_current,
                    'Lagging_Current_Power_Factor': lagging_pf,
                    'Leading_Current_Power_Factor': leading_pf,
                    'NSM': nsm,
                    'WeekStatus': week_status,
                    'Load_Type': load_type
                }
                
                # Create DataFrame and process
                input_df = pd.DataFrame([input_data])
                input_processed = preprocess_data(input_df, is_training=False)
                
                # Make prediction
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
