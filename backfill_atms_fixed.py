import pandas as pd
import random

def main():
    tx = pd.read_csv('data/transactions.csv')
    master = pd.read_csv('data/entity_master.csv')
    
    atm_tx = tx[tx['receiver_entity_id'].str.startswith('ATM_', na=False)]
    atms = atm_tx['receiver_entity_id'].unique()

    atm_states = {}
    for atm in atms:
        state = atm_tx[atm_tx['receiver_entity_id'] == atm]['receiver_state'].iloc[0]
        atm_states[atm] = state

    new_master = []
    new_loc = []

    for atm, state in atm_states.items():
        if atm not in master['entity_id'].values:
            new_master.append({
                'entity_id': atm,
                'account_number': 'N/A',
                'ifsc': 'N/A',
                'canonical_name': f'ATM Terminal {atm.split("_")[-1]}',
                'identity_key': atm
            })
            new_loc.append({
                'entity_id': atm,
                'state': state,
                'city': f"Unknown City ({state})", # Obvious fallback
                'latitude': 0.0,
                'longitude': 0.0
            })

    if new_master:
        df_m = pd.DataFrame(new_master)
        df_m = df_m[['entity_id', 'account_number', 'ifsc', 'canonical_name', 'identity_key']]
        df_m.to_csv('data/entity_master.csv', mode='a', header=False, index=False)
        
        df_l = pd.DataFrame(new_loc)
        df_l = df_l[['entity_id', 'state', 'city', 'latitude', 'longitude']]
        df_l.to_csv('data/entity_locations.csv', mode='a', header=False, index=False)
        
        print(f"Backfilled {len(new_master)} ATMs with explicit Unknown City labels.")

if __name__ == '__main__':
    main()
