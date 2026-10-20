import json
from src.models.threshold_tuning import tune_threshold

report = tune_threshold()
print(json.dumps(report, indent=2))