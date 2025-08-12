import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings

# Suppress a harmless scikit-learn warning that can occur with small datasets
warnings.filterwarnings('ignore', category=UserWarning)

def find_missing_data(df):
    """
    Flags rows with missing values and returns the updated DataFrame.
    """
    df['missing_data_flag'] = df.isnull().any(axis=1)
    missing_info = df[df['missing_data_flag']].apply(
        lambda row: ', '.join(df.columns[row.isnull()]), axis=1
    )
    df['missing_data_details'] = np.nan
    df.loc[df['missing_data_flag'], 'missing_data_details'] = missing_info
    return df

def detect_anomalies(df):
    """
    Uses an Isolation Forest model to detect anomalies (incorrect/outlier data).
    """
    numerical_cols = [
        'environment_score', 
        'social_score', 
        'governance_score', 
        'total_score'
    ]
    
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    
    if not numerical_cols_present:
        st.warning("No matching numerical columns found for anomaly detection. Skipping...")
        df['anomaly_flag'] = False
        return df
    
    df_filled = df[numerical_cols_present].fillna(df[numerical_cols_present].mean())
    
    model = IsolationForest(contamination='auto', random_state=42)
    df['anomaly_flag'] = model.fit_predict(df_filled)
    df['anomaly_flag'] = df['anomaly_flag'] == -1
    return df

def run_analysis(df):
    """
    Runs the full analysis pipeline on the provided DataFrame.
    """
    st.info("Running missing data detection...")
    df = find_missing_data(df)
    
    st.info("Running anomaly detection with Isolation Forest...")
    df = detect_anomalies(df)
    
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="ESG Data Anomaly Detector", layout="wide")
st.title("ESG Data Anomaly Detector")
st.write("Upload a CSV file to automatically detect missing values and statistical anomalies using an Isolation Forest model.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the uploaded CSV file into a DataFrame
        esg_data = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
        st.subheader("Original Data Preview")
        st.dataframe(esg_data.head())

        # Create a button to trigger the analysis
        if st.button("Start Analysis"):
            with st.spinner("Analyzing data..."):
                processed_data = run_analysis(esg_data.copy()) # Use a copy to avoid modifying original df
                flagged_data = processed_data[processed_data['missing_data_flag'] | processed_data['anomaly_flag']]

            if not flagged_data.empty:
                st.subheader("🚨 Detected Missing or Anomalous Data 🚨")
                st.dataframe(flagged_data)
                
                st.subheader("Summary of Anomalies")
                for index, row in flagged_data.iterrows():
                    company_id = row.get('ticker', row.get('company_id', 'N/A'))
                    if row['anomaly_flag']:
                        st.write(f"  - **Company '{company_id}'** flagged as an anomaly.")
                    if row['missing_data_flag']:
                        st.write(f"  - **Company '{company_id}'** is missing data in: `{row['missing_data_details']}`")
            else:
                st.success("✅ No missing or anomalous data was detected in the dataset.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
