import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

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
    """Load and perform initial data processing"""
    df = pd.read_csv("Steel_industry_data.csv")
    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    df.set_index('date', inplace=True)
    return df

def preprocess_features(df):
    """Preprocess features with explicit type handling"""
    df = df.copy()
    
    # Handle categorical variables first
    categorical_cols = ['WeekStatus', 'Load_Type']
    label_encoders = {}
    
    # Label encode categorical variables
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col]).astype(np.float64)
    
    # Extract datetime features
    df['year'] = df.index.year.astype(np.float64)
    df['month'] = df.index.month.astype(np.float64)
    df['day'] = df.index.day.astype(np.float64)
    df['hour'] = df.index.hour.astype(np.float64)
    df['dayofweek'] = df.index.dayofweek.astype(np.float64)
    
    # Ensure all numeric columns are float64
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)
    
    return df

def train_model(df):
    """Train the XGBoost model with preprocessed data"""
    # Preprocess features
    df_processed = preprocess_features(df)
    
    # Split into training and testing sets
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    # Prepare features and target
    X = trainer.drop(columns=['CO2(tCO2)'])
    y = trainer['CO2(tCO2)'].astype(np.float64)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = tts(
        X, y,
        train_size=0.8,
        random_state=42,
        shuffle=False
    )
    
    # Initialize and train model
    model = xgb.XGBRegressor(
        n_estimators=25,
        learning_rate=0.1,
        max_depth=7,
        subsample=1.0,
        random_state=42
    )
    
    # Train the model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model, tester

def plot_predictions(tester, model):
    """Plot actual vs predicted values"""
    X_test = tester.drop(columns=['CO2(tCO2)'])
    predictions = model.predict(X_test)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tester.index,
        y=tester['CO2(tCO2)'],
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
        marker=dict(color="#00ffcc")
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#00ffcc")
    )
    
    return fig

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    try:
        # Load data
        df = load_data()
        
        # Training section
        st.header("Model Training")
        if st.button("Train Model"):
            with st.spinner("Training in progress... 🔄"):
                model, test_data = train_model(df)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = test_data.drop(columns=['CO2(tCO2)']).columns.tolist()
                
                st.success("Model trained successfully! 🎉")
        
        # Visualization section
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
                    st.session_state['model'],
                    st.session_state['feature_names']
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