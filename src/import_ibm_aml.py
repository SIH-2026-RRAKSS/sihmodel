import pandas as pd
import networkx as nx
import os

def load_ibm_to_networkx(ibm_csv_path: str = "data/HI-Small_Trans.csv") -> nx.MultiDiGraph:
    """
    Parses the IBM AML dataset and converts it into the exact MultiDiGraph 
    schema expected by our Temporal BFS extractor.
    """
    print(f"Loading IBM AML Dataset from {ibm_csv_path}...")
    
    # The IBM dataset uses standard columns: 
    # Timestamp, From Bank, Account, To Bank, Account.1, Amount Paid, Is Laundering
    df = pd.read_csv(ibm_csv_path)
    
    # 1. Create unique Entity IDs across banks
    # e.g., "BankA_Account123"
    df['src_id'] = df['From Bank'].astype(str) + "_" + df['Account'].astype(str)
    df['dst_id'] = df['To Bank'].astype(str) + "_" + df['Account.1'].astype(str)
    
    # Sort chronologically to ensure time-series integrity
    df = df.sort_values(by=['Timestamp'])
    
    # 2. Initialize the Global Graph
    G = nx.MultiDiGraph()
    
    print("Populating graph nodes and edges...")
    
    # Clean column names dynamically to work with itertuples
    df.columns = df.columns.str.replace(' ', '_').str.replace('.', '_')
                  
    for row in df.itertuples(index=False):
        # IBM's 'Timestamp' is formatted like 'YYYY/MM/DD HH:MM'
        # Convert to unix epoch integer for fast numeric comparison in our Temporal BFS
        tx_time = int(pd.to_datetime(row.Timestamp).timestamp())
        
        # d16 is our pipeline's expected flag for 'Is Laundering'
        is_illicit = int(row.Is_Laundering)
        
        src = str(row.src_id)
        dst = str(row.dst_id)
        
        amount = float(row.Amount_Paid)
        currency = row.Payment_Currency
        
        G.add_edge(
            src, 
            dst, 
            amount=amount,
            timestamp=tx_time,
            d16=is_illicit,
            currency=currency
        )
        
        # Ensure nodes exist with basic metadata
        if not G.has_node(src):
            G.nodes[src]['node_type'] = 'account'
        if not G.has_node(dst):
            G.nodes[dst]['node_type'] = 'account'

    print(f"IBM Graph Loaded! Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    return G, df

def extract_ibm_incidents(G: nx.MultiDiGraph, df: pd.DataFrame, output_dir: str = "data/graphs_ibm/"):
    """
    Finds known laundering incidents in the IBM data and uses our 
    Temporal BFS to extract the chronological mule chains.
    """
    from graph_construction import extract_temporal_subgraph  # Import your perfected BFS
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all transactions flagged as laundering to use as root seeds
    illicit_txs = df[df['Is_Laundering'] == 1]
    
    # Group by the source account to find distinct laundering incidents
    incident_seeds = illicit_txs.groupby('src_id')['Timestamp'].min().reset_index()
    incident_seeds['label'] = 1
    
    # Sample negative (normal) transactions
    normal_txs = df[df['Is_Laundering'] == 0]
    # Sample 10x as many negative incidents to maintain a realistic base rate
    num_negatives = len(incident_seeds) * 10
    normal_seeds = normal_txs.sample(n=min(num_negatives, len(normal_txs)), random_state=42)
    normal_seeds = normal_seeds[['src_id', 'Timestamp']].rename(columns={'Timestamp': 'Timestamp'})
    normal_seeds['label'] = 0
    
    all_seeds = pd.concat([incident_seeds, normal_seeds]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Extracting {len(all_seeds)} IBM incident subgraphs ({len(incident_seeds)} Pos, {len(normal_seeds)} Neg) using Temporal BFS...")
    
    graph_count = 0
    for _, row in all_seeds.iterrows():
        seed_node = row['src_id']
        incident_time = int(pd.to_datetime(row['Timestamp']).timestamp())
        label = row['label']
        
        # EXACT SAME LOGIC: Max 72 hours, Top 15 degree cap, forward-time only
        subgraph = extract_temporal_subgraph(
            global_graph=G,
            root_node=seed_node,
            incident_timestamp=incident_time,
            max_hops=3,
            max_degree_per_hop=15
        )
        
        # Only save subgraphs that actually caught a multi-hop flow
        if subgraph.number_of_edges() > 1:
            if label == 1:
                graph_id = f"IBM_POS_{graph_count:05d}"
            else:
                graph_id = f"IBM_NEG_{graph_count:05d}"
                
            # Tag the root node so the GNN knows it's the incident center
            subgraph.nodes[seed_node]['is_incident'] = 1 
            subgraph.graph['contains_suspicious_activity'] = label
            
            nx.write_graphml(subgraph, f"{output_dir}/{graph_id}.graphml")
            graph_count += 1
            
    print(f"Successfully extracted {graph_count} operational subgraphs from IBM data.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Ingest IBM AML Dataset')
    parser.add_argument('--csv-path', type=str, default='data/HI-Small_Trans.csv', help='Path to HI-Small_Trans.csv')
    args = parser.parse_args()
    
    ibm_global_graph, ibm_df = load_ibm_to_networkx(ibm_csv_path=args.csv_path)
    extract_ibm_incidents(ibm_global_graph, ibm_df)
