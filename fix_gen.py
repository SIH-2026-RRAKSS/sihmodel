import os

path = 'src/generate_transactions.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Add saving of atm_lookup to df_locations
patch = """    df_locations, location_lookup = generate_entity_locations(entities, rng)
    atm_lookup = generate_atm_nodes(rng)
    
    # ADD ATMs to df_locations so they are saved
    atm_records = []
    for atm_id, atm_data in atm_lookup.items():
        atm_records.append({
            "entity_id": atm_id,
            "latitude": atm_data["latitude"],
            "longitude": atm_data["longitude"]
        })
    df_atms = pd.DataFrame(atm_records)
    df_locations = pd.concat([df_locations, df_atms], ignore_index=True)
"""
c = c.replace(
    'df_locations, location_lookup = generate_entity_locations(entities, rng)\n    atm_lookup = generate_atm_nodes(rng)',
    patch
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

print('Patched generate_transactions.py')
