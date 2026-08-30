import torch
import pandas as pd
from pathlib import Path
from src.graphsage_classifier import DualHeadGraphSAGE, build_pyg_data_from_graphml
from src.streaming_engine import normalize_single_graph_features
import torch.nn.functional as F

def main():
    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64)
    model.load_state_dict(torch.load('models/graphsage_model.pt', map_location='cpu', weights_only=True))
    model.eval()

    mean = torch.load('models/synthetic_mean.pt', map_location='cpu', weights_only=True)
    std = torch.load('models/synthetic_std.pt', map_location='cpu', weights_only=True)

    summary = pd.read_csv('data/graph_summary.csv')
    tp_cases = summary[(summary['contains_suspicious_activity'] == 1) & (summary['num_nodes'] > 15)]

    case = tp_cases.iloc[0]
    c_id = case['complaint_id']
    e_id = case['incident_entity_id']
    data = build_pyg_data_from_graphml(Path(f'data/graphs/{c_id}.graphml'), c_id, 1.0, e_id)
    
    x_norm = normalize_single_graph_features(data.x, mean, std)
    batch = torch.zeros(x_norm.size(0), dtype=torch.long)
    
    with torch.no_grad():
        x_conv1 = F.relu(model.conv1(x_norm, data.edge_index))
        node_embs = F.relu(model.conv2(x_conv1, data.edge_index))
        
        # Reversed
        rev_edges = data.edge_index[[1, 0]]
        x_conv1_rev = F.relu(model.conv1(x_norm, rev_edges))
        node_embs_rev = F.relu(model.conv2(x_conv1_rev, rev_edges))
        
        root_idx = (data.x[:, 10] == 1.0).nonzero(as_tuple=True)[0]
        if len(root_idx) > 0:
            root_idx = root_idx[0].item()
            
            root_emb_norm = node_embs[root_idx]
            root_emb_rev = node_embs_rev[root_idx]
            
            diff = torch.norm(root_emb_norm - root_emb_rev).item()
            print(f"Complaint: {c_id}")
            print(f"  Root Node Embedding Difference (L2 norm): {diff:.4f}")
            print(f"  Root Node Norm vs Rev Mean Absolute Diff: {torch.abs(root_emb_norm - root_emb_rev).mean().item():.4f}")
            
            prob_root_norm = torch.sigmoid(model.graph_classifier(root_emb_norm.unsqueeze(0))).item()
            prob_root_rev = torch.sigmoid(model.graph_classifier(root_emb_rev.unsqueeze(0))).item()
            print(f"  Root-Only Readout - Normal: {prob_root_norm:.4f} | Reversed: {prob_root_rev:.4f}")

if __name__ == '__main__':
    main()
