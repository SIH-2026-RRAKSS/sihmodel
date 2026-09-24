import pandas as pd
from src.streaming_engine import TemporalTransactionGraph

df = pd.read_csv("data/transactions.csv")
txs = df.to_dict("records")
engine = TemporalTransactionGraph(window_hours=72, max_hops=2)
for idx, tx in enumerate(txs[:2000]):
    _, triggered, reason, res = engine.ingest_transaction(tx)
    if res and res.get("risk_probability", 0) > 0.01:
        print(idx, triggered, res["risk_probability"])
print("Done")
