import streamlit as st
from joblib import load
import logging
import traceback
import pandas as pd

# Logging setup
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
def load_model(model_path):
    try:
        model = load(model_path)
        return model
    except FileNotFoundError:
        logging.error("Model file not found. Please provide the correct file path.")
    except Exception as e:
        logging.error("An error occurred while loading the model:")
        logging.error(traceback.format_exc())
        return None

# Streamlit app
def main():
    st.title("Credit Risk Prediction")

    # Prediction section
    st.header("Make Predictions")
    Age = st.number_input("Enter Age:", min_value=0.0)
    Income = st.number_input("Enter Income:", min_value=0.0)
    Emp_length = st.number_input("Enter Employment Length:", min_value=0.0)
    Amount = st.number_input("Enter Loan Amount:", min_value=0.0)
    Rate = st.number_input("Enter Interest Rate:", min_value=0.0)
    Percent_income = st.number_input("Enter Percent of Income:", min_value=0.0)
    Cred_length = st.number_input("Enter Credit Length:", min_value=0.0)
    Home = st.selectbox("Enter Home Ownership:", ['RENT', 'OWN', 'MORTGAGE'])
    Intent = st.selectbox("Enter Loan Intent:", ['EDUCATION', 'MEDICAL', 'DEBT_CONSOLIDATION', 'OTHER'])

    if st.button("Predict Credit Risk"):
        user_data = pd.DataFrame({
            'Age': [Age],
            'Income': [Income],
            'Emp_length': [Emp_length],
            'Amount': [Amount],
            'Rate': [Rate],
            'Percent_income': [Percent_income],
            'Cred_length': [Cred_length],
            'Home': [Home],
            'Intent': [Intent]
        })

        model_load_path = 'trained_xgboost_model.joblib'  # Hardcoded path for simplicity
        model = load_model(model_load_path)
        if model is not None and user_data is not None:
            try:
                prediction = model.predict(user_data)
                if prediction[0] == 1:
                    st.error("The individual is predicted to default on the loan.")
                else:
                    st.success("The individual is predicted to not default on the loan.")
            except Exception as e:
                logging.error("An error occurred during prediction:")
                logging.error(traceback.format_exc())

if __name__ == '__main__':
    main()
