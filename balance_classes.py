import pandas as pd
from sklearn.utils import resample

def balance_classes(df, target_column):
    # Map 'Y' to 1 and 'N' to 0 in the target column
    df[target_column] = df[target_column].map({'Y': 1, 'N': 0})

    # Separate majority and minority classes
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]

    # Get the size of the majority and minority classes
    majority_size = len(df_majority)
    minority_size = len(df_minority)

    if majority_size > minority_size:
        # Undersample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample without replacement
                                           n_samples=minority_size,  # to match minority class
                                           random_state=42)  # reproducible results

        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
    elif minority_size > majority_size:
        # Oversample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=majority_size,  # to match majority class
                                         random_state=42)  # reproducible results

        # Combine majority class with upsampled minority class
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
    else:
        # Classes are already balanced
        df_balanced = df

    # Map 1 back to 'Y' and 0 back to 'N'
    df_balanced[target_column] = df_balanced[target_column].map({1: 'Y', 0: 'N'})

    return df_balanced

# Example usage:
dataset_path = 'credit_risk.csv/credit_risk.csv'
df = pd.read_csv(dataset_path)

# Assuming 'Default' column is the target
df_balanced = balance_classes(df, target_column='Default')

# Save the balanced dataset (optional)
balanced_dataset_path = 'credit_risk.csv/credit_risk_balanced.csv'
df_balanced.to_csv(balanced_dataset_path, index=False)

# Check the balance of classes
print(df_balanced['Default'].value_counts())
