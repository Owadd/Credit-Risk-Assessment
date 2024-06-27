# 🎯 Credit Risk Assessment

Welcome to the Credit Risk Assessment project! 🎉 This repository is your one-stop-shop for predicting credit risk using the mighty power of the XGBoost classifier (and a few other models I threw in for fun).

## 📂 What's Inside?

### 1. `balance_classes.py`
Balance those pesky imbalanced classes like a pro! This script ensures that your dataset has a fair fight between the 'Y' and 'N' classes.

```python
import pandas as pd
from sklearn.utils import resample

def balance_classes(df, target_column):
    # Map 'Y' to 1 and 'N' to 0 in the target column
    df[target_column] = df[target_column].map({'Y': 1, 'N': 0})

    # Separate majority and minority classes
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]

    # Resample to balance classes
    if len(df_majority) > len(df_minority):
        df_majority = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
    else:
        df_minority = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)

    df_balanced = pd.concat([df_majority, df_minority])
    df_balanced[target_column] = df_balanced[target_column].map({1: 'Y', 0: 'N'})
    
    return df_balanced

# Example usage
dataset_path = 'credit_risk.csv/credit_risk.csv'
df = pd.read_csv(dataset_path)
df_balanced = balance_classes(df, target_column='Default')
df_balanced.to_csv('credit_risk_balanced.csv', index=False)
```

### 2. `combined_classifiers.py`
Why have one classifier when you can have six? 🎩✨ This script trains an ensemble of classifiers to boost your credit risk predictions.

```python
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

# [Function definitions for setup_logging, preprocess_data, and train_ensemble_classifier]

# Example usage
dataset_path = 'credit_risk.csv/new_training_data.csv'
model_save_path = 'trained_ensemble_model.joblib'
train_ensemble_classifier(dataset_path, model_save_path)
```

### 3. `credit_risk_app.py`
Streamlit app? Yes, please! 🌐 This script turns your credit risk model into an interactive web app.

```python
import streamlit as st
from joblib import load
import logging
import traceback
import pandas as pd

# [Function definitions for setup_logging, load_model, and main]

# Run the app
if __name__ == '__main__':
    main()
```

### 4. `credit_risk_evaluator.py`
Evaluate and predict with style. This script lets you train the XGBoost model, load it, and make predictions based on user input.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
import logging
import traceback
import xgboost as xgb

# [Function definitions for setup_logging, preprocess_data, train_xgboost_classifier, load_model, get_user_input, and predict_credit_risk]

# Example usage
dataset_path = 'credit_risk.csv/credit_risk_balanced.csv'
model_save_path = 'trained_xgboost_model.joblib'
train_xgboost_classifier(dataset_path, model_save_path)

model = load_model(model_save_path)
user_data = get_user_input()
predict_credit_risk(model, user_data)
```

### 5. `dataset_splitter.py`
Splits your dataset into training and testing sets because sharing is caring. 🍰

```python
import pandas as pd

# [Code for splitting the dataset]

# Example usage
data_file = "credit_risk.csv/credit_risk_balanced.csv"
# [Rest of the code]
```

### 6. `xg_boost_classifier_training.py`
Train the XGBoost model to be the best it can be. 🏋️‍♂️

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from joblib import dump
import logging
import traceback
import xgboost as xgb

# [Function definitions for setup_logging, preprocess_data, and train_xgboost_classifier]

# Example usage
dataset_path = 'credit_risk.csv/credit_risk_balanced.csv'
model_save_path = 'trained_xgboost_model.joblib'
train_xgboost_classifier(dataset_path, model_save_path)
```

## 🛠️ Tools & Tech

- **XGBoost**: The star of the show. 🌟
- **Streamlit**: Turns your scripts into interactive web apps.
- **Pandas**: Because data manipulation should be easy.
- **Scikit-learn**: Machine learning for everyone!
- **Joblib**: Save and load your models like a pro.

## 🎉 How to Use

1. **Clone the repo**: `git clone https://github.com/yourusername/CreditRiskAssessment.git`
2. **Navigate to the project directory**: `cd CreditRiskAssessment`
3. **Install the requirements**: `pip install -r requirements.txt`
4. **Balance your dataset**: Run `balance_classes.py`
5. **Train your model**: Run `xg_boost_classifier_training.py`
6. **Launch the app**: Run `streamlit run credit_risk_app.py`

## 🚀 Example

Run `credit_risk_evaluator.py` to train your XGBoost model, input your data, and predict credit risk with style.

## 💡 Notes

- This project was built with the XGBoost classifier. Any other models you see are just me experimenting! 🧪
- Feel free to play around with the code and make it your own. Contributions are welcome!

---

Enjoy predicting credit risk and may the XGBoost be with you! 🚀
