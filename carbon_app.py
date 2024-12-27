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
    
    # Drop Day_of_week column as we'll calculate it from the index
    if 'Day_of_week' in df.columns:
        df = df.drop(columns=['Day_of_week'])
    
    # Convert categorical variables to numeric using pd.get_dummies
    categorical_cols = ['WeekStatus', 'Load_Type']
    df = pd.get_dummies(df, columns=categorical_cols, dtype=np.float64)
    
    # Add time-based features
    df["year"] = df.index.year.astype(np.float64)
    df["month"] = df.index.month.astype(np.float64)
    df["dayofweek"] = df.index.dayofweek.astype(np.float64)
    df["day"] = df.index.day.astype(np.float64)
    df["hour"] = df.index.hour.astype(np.float64)
    
    return df

def train_model(df):
    """Train the XGBoost model"""
    df_processed = preprocess_data(df)
    
    # Split into training and testing
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
    
    return model, df_processed.columns

def plot_predictions(dates, predictions, actual):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name="Actual",
        line=dict(color="#00ffcc", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
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
                model, columns = train_model(df)
                st.session_state['model'] = model
                st.session_state['columns'] = columns
                st.success("Model trained successfully! 🎉")
        
        # Prediction Interface
        if 'model' in st.session_state:
            st.header("Predict CO2 Emissions")
            
            # Date and Time inputs
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Select Date", value=datetime.now())
            with col2:
                time_input = st.time_input("Select Time", value=time(12, 0))
            
            if st.button("Predict CO2 Emissions"):
                # Combine date and time
                datetime_input = datetime.combine(date, time_input)
                
                # Create a one-row DataFrame with the same structure as training data
                input_df = pd.DataFrame(index=[datetime_input])
                
                # Add required columns with average values from training data
                for col in df.columns:
                    if col != 'CO2(tCO2)':
                        input_df[col] = df[col].mean()
                
                # Process the input data the same way as training data
                input_processed = preprocess_data(input_df)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_processed)[0]
                
                # Display prediction
                st.success(f"Predicted CO2 Emissions for {datetime_input}: {prediction:.2f} tCO2")
                
                # Plot recent history with prediction
                recent_data = df.last('7D')  # Last 7 days of data
                recent_processed = preprocess_data(recent_data)
                recent_predictions = st.session_state['model'].predict(recent_processed.drop(columns=["CO2(tCO2)"]))
                
                fig = plot_predictions(
                    dates=recent_data.index,
                    predictions=recent_predictions,
                    actual=recent_data["CO2(tCO2)"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the data file is in the correct location and format.")

if __name__ == "__main__":
    main()