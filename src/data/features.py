
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction risk features without using the fraud label."""

    required = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()

    # Feature 1: Sender's actual balance reduction.
    result["sender_balance_change"] = (
        result["oldbalanceOrg"] - result["newbalanceOrig"]
    )

    # Feature 2: Receiver's actual balance increase.
    result["receiver_balance_change"] = (
        result["newbalanceDest"] - result["oldbalanceDest"]
    )

    # Feature 3: Difference between amount and sender balance reduction.
    result["sender_error"] = (
        result["sender_balance_change"] - result["amount"]
    )

    # Feature 4: Difference between amount and receiver balance increase.
    result["receiver_error"] = (
        result["receiver_balance_change"] - result["amount"]
    )

    # Feature 5: Transaction amount on a log scale.
    result["log_amount"] = np.log1p(result["amount"].clip(lower=0))

    # Feature 6: Flag whether the transaction amount is zero.
    result["is_zero_amount"] = (result["amount"] == 0).astype("int8")

    # Feature 7: Is this one of the transaction types associated with
    # fraud in our exploratory analysis?
    result["is_transfer_or_cashout"] = (
        result["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype("int8")

    # Do not drop or create target-derived information here.
    return result
