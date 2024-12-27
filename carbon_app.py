import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go

# Set page config for dark theme
st.set_page_config(
    page_title="Air Quality CO Predictor",
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
    """Handle data preprocessing"""
    df = df.copy()
    for col in df.columns:
        if col != 'date':
            df[col] = df[col].astype(np.float64)
    return df

def train_model(df):
    """Train the XGBoost model"""
    df_processed = preprocess_data(df)
    
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    X = trainer.drop(columns=["CO(GT)"])
    y = trainer["CO(GT)"].astype(np.float64)
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

# [Previous plotting functions remain the same]
def plot_predictions(tester, model):
    """Plot actual vs predicted values"""
    X_test = tester.drop(columns=["CO(GT)"])
    predictions = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester["CO(GT)"],
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
        title="CO Levels: Actual vs Predicted",
        xaxis_title="Sample",
        yaxis_title="CO Level",
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
    st.title("🌍 Air Quality Predictor")
    
    try:
        # Load data
        df = pd.read_csv("Steel_industry_data.csv")
        
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
            st.header("Predict CO Level")
            
            # Feature sliders with statistics-based ranges
            st.subheader("Adjust Sensor Readings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pt08_s1 = st.slider("PT08.S1(CO)", 
                                  min_value=921.0,  # 25th percentile
                                  max_value=1221.0,  # 75th percentile
                                  value=1053.0)  # median
                                  
                pt08_s2 = st.slider("PT08.S2(NMHC)",
                                  min_value=711.0,
                                  max_value=1105.0,
                                  value=895.0)
                                  
                pt08_s3 = st.slider("PT08.S3(NOx)",
                                  min_value=637.0,
                                  max_value=960.0,
                                  value=794.0)
                                  
                temp = st.slider("Temperature (T)",
                               min_value=10.9,
                               max_value=24.1,
                               value=17.2)
            
            with col2:
                nmhc = st.slider("NMHC(GT)",
                               min_value=-200.0,
                               max_value=1189.0,
                               value=-200.0)
                               
                nox = st.slider("NOx(GT)",
                              min_value=50.0,
                              max_value=284.0,
                              value=141.0)
                              
                no2 = st.slider("NO2(GT)",
                              min_value=53.0,
                              max_value=133.0,
                              value=96.0)
                              
                rh = st.slider("Relative Humidity (RH)",
                             min_value=34.1,
                             max_value=61.9,
                             value=48.6)
            
            with col3:
                pt08_s4 = st.slider("PT08.S4(NO2)",
                                  min_value=1185.0,
                                  max_value=1662.0,
                                  value=1446.0)
                                  
                pt08_s5 = st.slider("PT08.S5(O3)",
                                  min_value=700.0,
                                  max_value=1255.0,
                                  value=942.0)
                                  
                ah = st.slider("Absolute Humidity (AH)",
                             min_value=0.6923,
                             max_value=1.2962,
                             value=0.9768,
                             step=0.0001)
            
            if st.button("Predict CO Level"):
                # Create prediction input
                input_data = {
                    'PT08.S1(CO)': pt08_s1,
                    'NMHC(GT)': nmhc,
                    'PT08.S2(NMHC)': pt08_s2,
                    'NOx(GT)': nox,
                    'PT08.S3(NOx)': pt08_s3,
                    'NO2(GT)': no2,
                    'PT08.S4(NO2)': pt08_s4,
                    'PT08.S5(O3)': pt08_s5,
                    'T': temp,
                    'RH': rh,
                    'AH': ah
                }
                
                # Create DataFrame and process
                input_df = pd.DataFrame([input_data])
                input_processed = preprocess_data(input_df)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_processed)[0]
                st.success(f"Predicted CO Level: {prediction:.2f}")
            
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