import pandas as pd

def main():
    summary = pd.read_csv('data/graph_summary.csv')
    tx = pd.read_csv('data/transactions.csv')
    
    cases = ['C000014', 'C000016']
    for c_id in cases:
        row = summary[summary['complaint_id'] == c_id].iloc[0]
        # the graph extracts window between window_start and window_end
        # but to keep it simple, let's just check the incident_entity's in-degree and the subgraph's max transaction
        inc_ent = row['incident_entity_id']
        start = row['window_start']
        end = row['window_end']
        
        # filter tx in window
        mask = (tx['timestamp'] >= start) & (tx['timestamp'] <= end)
        tx_window = tx[mask]
        
        # for max_transaction_value, we need to know all edges in the graph. 
        # let's just use the python NetworkX graph that was extracted.
        import networkx as nx
        from pathlib import Path
        G = nx.read_graphml(Path(f'data/graphs/{c_id}.graphml'))
        
        # calculate max tx from graph
        max_tx = 0.0
        in_deg_inc = G.in_degree(inc_ent)
        
        for u, v, d in G.edges(data=True):
            amt = float(d.get('amount', 0))
            if amt > max_tx:
                max_tx = amt
                
        print(f"Complaint: {c_id}")
        print(f"  Max TX (Graph): {max_tx:.2f} | Max TX (CSV): {row['max_transaction_value']:.2f}")
        print(f"  In-Deg Inc (Graph): {in_deg_inc} | In-Deg Inc (CSV): {row['in_degree_incident']}")

if __name__ == '__main__':
    main()
