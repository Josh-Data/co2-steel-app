# [Previous imports remain the same]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from datetime import datetime, time

# [Previous code remains the same until the main function]

def create_prediction_interface():
    """Create an interface for making predictions with date/time inputs"""
    st.header("Make a Prediction")
    
    # Date and Time inputs
    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Select Date", value=datetime.now())
    with col2:
        time_input = st.time_input("Select Time", value=time(12, 0))
    
    # Combine date and time
    datetime_input = datetime.combine(date, time_input)
    
    # Create all needed features for prediction
    features = {}
    
    # Time-based features
    features["year"] = float(datetime_input.year)
    features["month"] = float(datetime_input.month)
    features["day"] = float(datetime_input.day)
    features["hour"] = float(datetime_input.hour)
    features["dayofweek"] = float(datetime_input.weekday())
    
    # Other features with sliders
    col1, col2 = st.columns(2)
    with col1:
        features["Use_kWh"] = st.slider("Use kWh", min_value=0.0, max_value=100.0, value=50.0)
        features["Lagging_Current_Reactive.Power_kVarh"] = st.slider("Lagging Current", min_value=0.0, max_value=100.0, value=50.0)
        features["Leading_Current_Reactive_Power_kVarh"] = st.slider("Leading Current", min_value=0.0, max_value=100.0, value=50.0)
        features["WeekStatus"] = st.selectbox("Week Status", options=["Weekday", "Weekend"])
    
    with col2:
        features["CO2(tCO2)"] = st.slider("Expected CO2", min_value=0.0, max_value=100.0, value=50.0)
        features["Load_Type"] = st.selectbox("Load Type", options=["Light Load", "Medium Load", "Maximum Load"])
        
    return features

def predict_co2(model, features):
    """Make a prediction using the trained model"""
    # Convert categorical variables to dummy variables
    # WeekStatus
    features["WeekStatus_Weekday"] = 1.0 if features["WeekStatus"] == "Weekday" else 0.0
    features["WeekStatus_Weekend"] = 1.0 if features["WeekStatus"] == "Weekend" else 0.0
    
    # Load_Type
    features["Load_Type_Light Load"] = 1.0 if features["Load_Type"] == "Light Load" else 0.0
    features["Load_Type_Medium Load"] = 1.0 if features["Load_Type"] == "Medium Load" else 0.0
    features["Load_Type_Maximum Load"] = 1.0 if features["Load_Type"] == "Maximum Load" else 0.0
    
    # Remove original categorical columns
    del features["WeekStatus"]
    del features["Load_Type"]
    
    # Create DataFrame with the same columns as training data
    input_df = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    try:
        # Load data
        df = pd.read_csv("Steel_industry_data.csv", index_col="date")
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        
        # Training section
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training in progress... 🔄"):
                model, test_data = train_model(df)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                
                st.success("Model trained successfully! 🎉")
        
        # Prediction Interface
        if 'model' in st.session_state:
            features = create_prediction_interface()
            
            if st.button("Predict CO2 Emissions"):
                prediction = predict_co2(st.session_state['model'], features.copy())
                st.success(f"Predicted CO2 Emissions: {prediction:.2f} tCO2")
        
        # Visualization section [Previous visualization code remains the same]
        if 'model' in st.session_state:
            st.header("Model Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predictions vs Actual Values")
                pred_fig = plot_predictions(
                    st.session_state['test_data'],
                    st.session_state['model']
                )
                st.plotly_chart(pred_fig, use_container_width=True)
            
            with col2:
                st.subheader("Feature Importance")
                imp_fig = plot_feature_importance(
                    st.session_state['model']
                )
                st.plotly_chart(imp_fig, use_container_width=True)
            
            # Metrics
            X_test = st.session_state['test_data'].drop(columns=['CO2(tCO2)'])
            predictions = st.session_state['model'].predict(X_test)
            actual = st.session_state['test_data']['CO2(tCO2)']
            
            mse = np.mean((predictions - actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actual))
            
            st.header("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
            col3.metric("Mean Absolute Error", f"{mae:.2f}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the data file is in the correct location and format.")

if __name__ == "__main__":
    main()