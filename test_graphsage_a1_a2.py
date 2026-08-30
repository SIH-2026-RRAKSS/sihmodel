import torch
import pandas as pd
from pathlib import Path
from src.graphsage_classifier import DualHeadGraphSAGE, build_pyg_data_from_graphml
from src.streaming_engine import normalize_single_graph_features
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F

def main():
    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64)
    model.load_state_dict(torch.load('models/graphsage_model.pt', map_location='cpu', weights_only=True))
    model.eval()

    class DualHeadGraphSAGE_MaxPool(DualHeadGraphSAGE):
        def forward(self, x, edge_index, batch):
            x_conv1 = self.conv1(x, edge_index)
            x_relu1 = F.relu(x_conv1)
            x_drop = self.dropout(x_relu1)
            node_embeddings = F.relu(self.conv2(x_drop, edge_index))
            node_logits = self.node_classifier(node_embeddings)
            graph_embedding = global_max_pool(node_embeddings, batch)
            graph_logits = self.graph_classifier(graph_embedding)
            return node_logits.squeeze(-1), graph_logits.squeeze(-1), graph_embedding

    model_max = DualHeadGraphSAGE_MaxPool(input_dim=13, hidden_dim=64)
    model_max.load_state_dict(model.state_dict())
    model_max.eval()

    mean = torch.load('models/synthetic_mean.pt', map_location='cpu', weights_only=True)
    std = torch.load('models/synthetic_std.pt', map_location='cpu', weights_only=True)

    summary = pd.read_csv('data/graph_summary.csv')
    tp_cases = summary[(summary['contains_suspicious_activity'] == 1) & (summary['num_nodes'] > 15)]

    for i in range(2):
        case = tp_cases.iloc[i]
        c_id = case['complaint_id']
        e_id = case['incident_entity_id']
        data = build_pyg_data_from_graphml(Path(f'data/graphs/{c_id}.graphml'), c_id, 1.0, e_id)
        x_norm = normalize_single_graph_features(data.x, mean, std)
        batch = torch.zeros(x_norm.size(0), dtype=torch.long)
        
        with torch.no_grad():
            prob_norm = torch.sigmoid(model(x_norm, data.edge_index, batch)[1]).item()
            prob_norm_max = torch.sigmoid(model_max(x_norm, data.edge_index, batch)[1]).item()
            
            reversed_edges = data.edge_index[[1, 0]]
            prob_rev = torch.sigmoid(model(x_norm, reversed_edges, batch)[1]).item()
            prob_rev_max = torch.sigmoid(model_max(x_norm, reversed_edges, batch)[1]).item()
            
        print(f"Complaint: {c_id}")
        print(f"  Mean Pool - Normal: {prob_norm:.4f} | Reversed: {prob_rev:.4f}")
        print(f"  Max Pool  - Normal: {prob_norm_max:.4f} | Reversed: {prob_rev_max:.4f}")

if __name__ == '__main__':
    main()
