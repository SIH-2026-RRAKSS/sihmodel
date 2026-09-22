import sqlite3
import pandas as pd

conn = sqlite3.connect("sihmodel/data/cybercrime_aml.db")
df = pd.read_sql("""
    SELECT t.sender_entity_id as entityId, t.receiver_entity_id as atmId, t.amount,
           e1.district as fromCity, e1.latitude as from_lat, e1.longitude as from_lon,
           e2.district as toCity, e2.latitude as to_lat, e2.longitude as to_lon
    FROM transactions t
    JOIN entity_master e1 ON t.sender_entity_id = e1.entity_id
    JOIN entity_master e2 ON t.receiver_entity_id = e2.entity_id
    WHERE e1.district != e2.district AND e1.latitude IS NOT NULL AND e2.latitude IS NOT NULL
    LIMIT 20
""", conn)
print(df.to_dict(orient="records"))
