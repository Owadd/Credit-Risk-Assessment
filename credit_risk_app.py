import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
import logging
import traceback
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Logging setup
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data preprocessing
def preprocess_data(df):
    df = df.dropna(subset=['Default'])
    numerical_cols = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    categorical_cols = ['Home', 'Intent']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    df['Default'] = df['Default'].map({'Y': 1, 'N': 0})
    return df

# Training XGBoost classifier
def train_xgboost_classifier(dataset_path, model_save_path):
    try:
        setup_logging()
        logging.info("Starting the XGBoost training process.")
        data = pd.read_csv(dataset_path)
        logging.info("Preprocessing the dataset.")
        data = preprocess_data(data)

        X = data.drop(columns=['Id', 'Default', 'Status'])
        y = data['Default']

        logging.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numerical_features = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
        categorical_features = ['Home', 'Intent']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
        ])

        logging.info("Training the XGBoost model.")
        clf.fit(X_train, y_train)

        logging.info(f"Saving the trained model to {model_save_path}.")
        dump(clf, model_save_path)

        logging.info("Making predictions on the testing data.")
        y_pred = clf.predict(X_test)

        logging.info("Evaluating the model's performance.")
        st.text("XGBoost Classification Report:")
        st.text(classification_report(y_test, y_pred))

    except FileNotFoundError:
        logging.error("File not found. Please provide the correct file path.")
    except Exception as e:
        logging.error("An error occurred during training the XGBoost model:")
        logging.error(traceback.format_exc())

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

    # Model training section
    st.header("Train Model")
    dataset_path = st.text_input("Enter the path to the dataset:", './credit_risk.csv/credit_risk_balanced.csv')
    model_save_path = st.text_input("Enter the path to save the model:", 'trained_xgboost_model.joblib')
    
    if st.button("Train Model"):
        train_xgboost_classifier(dataset_path, model_save_path)
        st.success("Model training completed and saved.")

    # Load model section
    st.header("Load Model")
    model_load_path = st.text_input("Enter the path to load the model:", 'trained_xgboost_model.joblib')
    if st.button("Load Model"):
        model = load_model(model_load_path)
        if model:
            st.success("Model loaded successfully.")
        else:
            st.error("Error loading model.")

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
