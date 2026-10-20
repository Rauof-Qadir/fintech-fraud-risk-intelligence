import json
from src.data.cleaning import clean_dataset

report = clean_dataset()
print(json.dumps(report, indent=2))