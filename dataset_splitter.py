import pandas as pd

try:
    # Replace 'your_sentiment_file.csv' with the actual filename
    data_file = "credit_risk.csv/credit_risk_balanced.csv"

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(data_file)

    # Shuffle the DataFrame for randomness (important!)
    df = df.sample(frac=1)

    # Calculate split points (assuming a sentiment label column named 'sentiment')
    split_point = int(0.8 * len(df))
    train_data = df[0:split_point]
    test_data = df[split_point:]

    # Optional: Save the split data to separate CSV files
    train_data.to_csv("./credit_risk.csv/training_data.csv", index=False)
    test_data.to_csv("./credit_risk.csv/testing_data.csv", index=False)

    print("Data split complete!")
    print(f"Training data size: {len(train_data)}")
    print(f"Testing data size: {len(test_data)}")

except FileNotFoundError:
    print("Error: The file specified was not found.")
except pd.errors.EmptyDataError:
    print("Error: The specified file is empty.")
except Exception as e:
    print("An unexpected error occurred:", str(e))
