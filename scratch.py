import pandas as pd
df = pd.read_csv('data/entity_master.csv')
print(df['entity_id'].duplicated().sum())
