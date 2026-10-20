import json
from src.data.eda import run_eda

report = run_eda(
    csv_path="data/raw/transactions.csv",
    output_dir="reports/figures",
)

print(json.dumps(report, indent=2))