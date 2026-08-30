import sqlite3
import pandas as pd
conn = sqlite3.connect('data/cybercrime_aml.db')
pred = pd.read_sql_query("SELECT entity_id, district, state FROM entity_master WHERE entity_type='ATM' LIMIT 5", conn)
print(pred)
