import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
import plotly.graph_objects as go

# Set page config for dark theme
st.set_page_config(
    page_title="Steel Industry CO2 Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏭 Steel Industry CO2 Emissions Predictor")
    
    try:
        # Load data
        df = pd.read_csv("Steel_industry_data.csv")
        
        # Display initial data info
        st.write("Initial Data Types:", df.dtypes)
        st.write("Sample of Raw Data:", df.head())
        
        # Display columns
        st.write("Columns in dataset:", df.columns.tolist())
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the data file is in the correct location and format.")

if __name__ == "__main__":
    main()
    