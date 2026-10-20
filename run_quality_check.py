
import json

from src.data.quality import inspect_dataset


DATA_PATH = "data/raw/transactions.csv"

report = inspect_dataset(DATA_PATH)

print(json.dumps(report, indent=2))
