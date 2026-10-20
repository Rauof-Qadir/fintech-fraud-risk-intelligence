import json
from src.models.baseline import train_baseline

report = train_baseline()
print(json.dumps(report, indent=2))