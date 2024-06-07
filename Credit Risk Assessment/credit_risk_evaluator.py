import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
import logging
import traceback
import xgboost as xgb
import numpy as np

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df):
    df = df.dropna(subset=['Default'])
    numerical_cols = ['Age', 'Income', 'Emp_length', 'Amount', 'Rate', 'Percent_income', 'Cred_length']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    categorical_cols = ['Home', 'Intent']
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    df['Default'] = df['Default'].map({'Y': 1, 'N': 0})
    return df

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
        print("XGBoost Classification Report:")
        print(classification_report(y_test, y_pred))

    except FileNotFoundError:
        logging.error("File not found. Please provide the correct file path.")
    except Exception as e:
        logging.error("An error occurred during training the XGBoost model:")
        logging.error(traceback.format_exc())

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

def get_user_input():
    try:
        Age = float(input("Enter Age: "))
        Income = float(input("Enter Income: "))
        Emp_length = float(input("Enter Employment Length: "))
        Amount = float(input("Enter Loan Amount: "))
        Rate = float(input("Enter Interest Rate: "))
        Percent_income = float(input("Enter Percent of Income: "))
        Cred_length = float(input("Enter Credit Length: "))
        Home = input("Enter Home Ownership (RENT, OWN, MORTGAGE): ")
        Intent = input("Enter Loan Intent (EDUCATION, MEDICAL, etc.): ")
        
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
        
        return user_data
    except ValueError:
        logging.error("Invalid input. Please enter numeric values for numeric fields.")
        return None

def predict_credit_risk(model, user_data):
    try:
        if user_data is not None:
            prediction = model.predict(user_data)
            if prediction[0] == 1:
                print("The individual is predicted to default on the loan.")
            else:
                print("The individual is predicted to not default on the loan.")
        else:
            logging.error("User data is None. Prediction cannot be made.")
    except Exception as e:
        logging.error("An error occurred during prediction:")
        logging.error(traceback.format_exc())

# Example usage:
dataset_path = './credit_risk.csv/credit_risk_balanced.csv'
model_save_path = 'trained_xgboost_model.joblib'

# Train the model (uncomment if the model is not already trained)
# train_xgboost_classifier(dataset_path, model_save_path)

# Load the trained model
model = load_model(model_save_path)

# Get user input
user_data = get_user_input()

# Make a prediction
if model is not None and user_data is not None:
    predict_credit_risk(model, user_data)
