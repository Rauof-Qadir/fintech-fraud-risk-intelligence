import json
from src.models.business_cost import optimize_business_cost

report = optimize_business_cost()
print(json.dumps(report, indent=2))