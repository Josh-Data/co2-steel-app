import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from datetime import datetime, time

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

def preprocess_data(df):
    """Handle categorical variables and add time features"""
    df = df.copy()
    
    # Drop Day_of_week as we'll calculate it from the index
    if 'Day_of_week' in df.columns:
        df = df.drop(columns=['Day_of_week'])
    
    # Convert categorical variables to numeric
    # Clean categorical values by replacing spaces with underscores
    if 'Load_Type' in df.columns:
        df['Load_Type'] = df['Load_Type'].str.replace(' ', '_')
    
    categorical_cols = ['WeekStatus', 'Load_Type']
    df = pd.get_dummies(df, columns=categorical_cols, dtype=np.float64)
    
    # Add time-based features
    df["year"] = df.index.year.astype(np.float64)
    df["month"] = df.index.month.astype(np.float64)
    df["dayofweek"] = df.index.dayofweek.astype(np.float64)
    df["day"] = df.index.day.astype(np.float64)
    df["hour"] = df.index.hour.astype(np.float64)
    
    # Ensure all numeric columns are float64
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)
    
    return df

def train_model(df):
    """Train the XGBoost model"""
    df_processed = preprocess_data(df)
    
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
    
    try:
        # Load data
        df = pd.read_csv("Steel_industry_data.csv", index_col="date")
        df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
        
        # Training section
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training in progress... 🔄"):
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                st.success("Model trained successfully! 🎉")
        
        # Prediction Interface
        if 'model' in st.session_state:
            st.header("Predict CO2 Emissions")
            
            # Date and Time inputs (limited to 2018-2019)
            min_date = datetime(2018, 1, 1)
            max_date = datetime(2019, 12, 31)
            default_date = datetime(2018, 6, 1)
            
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input(
                    "Select Date",
                    value=default_date,
                    min_value=min_date,
                    max_value=max_date
                )
            with col2:
                time_input = st.time_input("Select Time", value=time(12, 0))
            
            # Feature sliders
            st.subheader("Adjust Features")
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
                
                week_status = st.selectbox("Week Status", options=["Weekday", "Weekend"])
                load_type = st.selectbox("Load Type", options=["Light_Load", "Medium_Load", "Maximum_Load"])
            
            if st.button("Predict CO2 Emissions"):
                # Create prediction input
                datetime_input = datetime.combine(date, time_input)
                input_data = {
                    'Usage_kWh': usage_kwh,
                    'Lagging_Current_Reactive.Power_kVarh': lagging_current,
                    'Leading_Current_Reactive_Power_kVarh': leading_current,
                    'Lagging_Current_Power_Factor': lagging_pf,
                    'Leading_Current_Power_Factor': leading_pf,
                    'NSM': datetime_input.hour * 3600 + datetime_input.minute * 60,
                    'WeekStatus': week_status,
                    'Load_Type': load_type
                }
                
                # Create DataFrame and process
                input_df = pd.DataFrame([input_data], index=[datetime_input])
                input_processed = preprocess_data(input_df)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_processed)[0]
                st.success(f"Predicted CO2 Emissions: {prediction:.2f} tCO2")
            
            # Show visualizations
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
                    st.session_state['model'],
                    st.session_state['feature_names']
                )
                st.plotly_chart(imp_fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the data file is in the correct location and format.")

if __name__ == "__main__":
    main()