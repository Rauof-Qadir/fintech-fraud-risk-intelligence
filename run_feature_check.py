
import pandas as pd
from src.data.features import engineer_features

df = pd.read_csv("data/processed/transactions_clean.csv", nrows=1000)

featured = engineer_features(df)

new_columns = [
    "sender_balance_change",
    "receiver_balance_change",
    "sender_error",
    "receiver_error",
    "log_amount",
    "is_zero_amount",
    "is_transfer_or_cashout",
]

print("Original shape:", df.shape)
print("New shape:", featured.shape)
print("\nSample engineered features:")
print(featured[new_columns].head(10).to_string(index=False))
print("\nMissing values in new features:")
print(featured[new_columns].isna().sum().to_dict())
