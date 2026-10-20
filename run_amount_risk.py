from src.data.amount_risk import analyze_amount_risk

report = analyze_amount_risk()
print(report.to_string(index=False))