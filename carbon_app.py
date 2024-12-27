import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

def main():
    st.title("Column Name Diagnostic")
    
    try:
        # Load data
        df = pd.read_csv("Steel_industry_data.csv")
        
        # Display exact column names
        st.write("### Exact Column Names:")
        for col in df.columns:
            st.write(f"'{col}'")
            
        # Display data types
        st.write("\n### Data Types:")
        st.write(df.dtypes)
        
        # Display first few rows
        st.write("\n### First Few Rows:")
        st.write(df.head())
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()