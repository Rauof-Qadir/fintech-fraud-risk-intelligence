import json
from src.data.splitting import create_train_test_split

report = create_train_test_split()
print(json.dumps(report, indent=2))