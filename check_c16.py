import sqlite3
import pandas as pd
conn = sqlite3.connect("data/cybercrime_aml.db")
pred = pd.read_sql_query("SELECT complaint_id, graphsage_risk_probability, confidence_tier FROM incident_predictions WHERE complaint_id='C000016'", conn)
print(pred)
