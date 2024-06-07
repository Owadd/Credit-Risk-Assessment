import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from joblib import dump
import logging
import traceback

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

def train_ensemble_classifier(dataset_path, model_save_path):
    try:
        setup_logging()
        logging.info("Starting the ensemble training process.")
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

        xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        rf_clf = RandomForestClassifier(random_state=42)
        lr_clf = LogisticRegression(random_state=42)
        gb_clf = GradientBoostingClassifier(random_state=42)
        ada_clf = AdaBoostClassifier(random_state=42)
        et_clf = ExtraTreesClassifier(random_state=42)

        ensemble_clf = VotingClassifier(estimators=[
            ('xgb', xgb_clf),
            ('rf', rf_clf),
            ('lr', lr_clf),
            ('gb', gb_clf),
            ('ada', ada_clf),
            ('et', et_clf)
        ], voting='soft')

        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', ensemble_clf)
        ])

        logging.info("Training the ensemble model.")
        clf.fit(X_train, y_train)

        logging.info(f"Saving the trained model to {model_save_path}.")
        dump(clf, model_save_path)

        logging.info("Making predictions on the testing data.")
        y_pred = clf.predict(X_test)

        logging.info("Evaluating the model's performance.")
        print("Ensemble (XGBoost + Random Forest + Logistic Regression + Gradient Boosting + AdaBoost + Extra Trees) Classification Report:")
        print(classification_report(y_test, y_pred))

    except FileNotFoundError:
        logging.error("File not found. Please provide the correct file path.")
    except Exception as e:
        logging.error("An error occurred during training the ensemble model:")
        logging.error(traceback.format_exc())

# Example usage:
dataset_path = './credit_risk.csv/new_training_data.csv'
model_save_path = 'trained_ensemble_model.joblib'
train_ensemble_classifier(dataset_path, model_save_path)
