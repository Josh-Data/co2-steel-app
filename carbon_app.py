import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go
from datetime import datetime, time
import os

# Oy vey, let's set up this fancy schmancy dark theme
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
    """Load and cache the dataset like your bubbe's secret recipe"""
    try:
        df = pd.read_csv("Steel_industry_data.csv")
        st.write("Original columns in dataset:", df.columns.tolist())
        return df, None
    except Exception as e:
        return None, str(e)

def preprocess_data(df, is_training=False):
    """
    Oy gevalt! Let's clean up this data like we're preparing for Pesach!
    """
    df = df.copy()
    
    # First, like a good baleboste, we'll clean house
    if 'Day_of_week' in df.columns:
        df = df.drop(columns=['Day_of_week'])
    
    # Handle the categorical variables like a maven
    if 'WeekStatus' in df.columns:
        weekstatus_dummies = pd.get_dummies(df['WeekStatus'], prefix='WeekStatus')
        df = pd.concat([df, weekstatus_dummies], axis=1)
        df = df.drop('WeekStatus', axis=1)
    
    if 'Load_Type' in df.columns:
        # Clean it up like your bubby's kitchen
        df['Load_Type'] = df['Load_Type'].str.strip().str.replace(' ', '_')
        loadtype_dummies = pd.get_dummies(df['Load_Type'], prefix='Load_Type')
        df = pd.concat([df, loadtype_dummies], axis=1)
        df = df.drop('Load_Type', axis=1)
    
    # Make everything float64, or your model will plotz!
    for col in df.columns:
        if col != 'CO2(tCO2)':  # Don't touch the target, it's shayna
            try:
                df[col] = df[col].astype(np.float64)
            except Exception as e:
                st.error(f"Oy vey! Problem with column {col}: {str(e)}")
                st.write(f"Column {col} unique values:", df[col].unique())
                raise
    
    return df

def train_model(df):
    """Train the model like you're teaching your kinderlach"""
    st.write("Training data columns:", df.columns.tolist())
    
    co2_column = 'CO2(tCO2)'
    if co2_column not in df.columns:
        st.error(f"Oy gevalt! No CO2 column found! We have these columns:")
        st.write(df.columns.tolist())
        raise ValueError(f"Missing CO2 column - such a tzimmes!")
    
    st.write(f"Using {co2_column} as target variable")
    
    # Keep the CO2 values kosher
    co2_values = df[co2_column].astype(np.float64)
    
    # Process the data like you're making gefilte fish
    df_processed = preprocess_data(df, is_training=True)
    st.write("Processed data columns:", df_processed.columns.tolist())
    
    # A bissel debugging information
    st.write("Data types after preprocessing:", df_processed.dtypes.to_dict())
    
    # Put the CO2 values back like the cherry on the babka
    df_processed[co2_column] = co2_values
    
    length = len(df_processed)
    main = int(length * 0.8)
    trainer = df_processed[:main]
    tester = df_processed[main:]
    
    X = trainer.drop(columns=[co2_column])
    y = trainer[co2_column]
    
    st.write("Feature columns for training:", X.columns.tolist())
    
    X_train, X_val, y_train, y_val = tts(X, y, train_size=0.8, random_state=42, shuffle=False)
    
    # Make a model that's a real mensch
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
    """Plot predictions like you're plotting shidduchim"""
    co2_column = 'CO2(tCO2)'
    if co2_column not in tester.columns:
        st.error(f"Oy vey iz mir! No CO2 column for plotting!")
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
        showlegend=True
    )
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance like reading the megillah"""
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
    
    # Load the data like you're unpacking your bubby's suitcase
    df, error = load_data()
    if error:
        st.error(f"Such tsuris! Error loading data: {error}")
        return
        
    if df is None:
        st.error("Oy vey, no data!")
        return
        
    st.success("Nu, the data loaded successfully!")
    st.write("Dataset Shape:", df.shape)
    st.write("First few rows of this mishegoss:")
    st.write(df.head())
    
    # Training section - like teaching your kinderlach to make challah
    st.header("Model Training")
    if st.button("Train Model"):
        with st.spinner("Training in progress... have some patience, bubbeleh! 🔄"):
            try:
                model, test_data, feature_names = train_model(df)
                st.session_state['model'] = model
                st.session_state['test_data'] = test_data
                st.session_state['feature_names'] = feature_names
                st.success("Mazel tov! Model trained successfully! 🎉")
            except Exception as e:
                st.error(f"Oy gevalt! Error during training: {str(e)}")
                return

    # Prediction Interface - where the magic happens
    if 'model' in st.session_state:
        st.header("Predict CO2 Emissions")
        
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
        
        # Categories, like choosing between kugel and tzimmes
        col1, col2 = st.columns(2)
        with col1:
            week_status = st.selectbox("Week Status", options=["Weekday", "Weekend"])
        with col2:
            load_type = st.selectbox("Load Type", options=["Light_Load", "Medium_Load", "Maximum_Load"])
        
        if st.button("Predict CO2 Emissions"):
            try:
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
                
                input_df = pd.DataFrame([input_data])
                input_processed = preprocess_data(input_df, is_training=False)
                
                prediction = st.session_state['model'].predict(input_processed)[0]
                st.success(f"Predicted CO2 Emissions: {prediction:.2f} tCO2")
                
            except Exception as e:
                st.error(f"Oy vey iz mir! Error during prediction: {str(e)}")
                st.write("Debug info:", input_processed.columns.tolist())
        
        # Show the plotzes (plots)
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