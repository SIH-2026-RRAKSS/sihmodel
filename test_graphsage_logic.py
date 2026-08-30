import torch
import pandas as pd
from pathlib import Path
from src.graphsage_classifier import DualHeadGraphSAGE, build_pyg_data_from_graphml
import random

def main():
    model_path = Path("models/graphsage_model.pt")
    if not model_path.exists():
        print("Model file not found!")
        return
        
    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    # Find a true positive graph
    summary = pd.read_csv("data/graph_summary.csv")
    tp_cases = summary[summary['contains_suspicious_activity'] == 1].head(1)
    if len(tp_cases) == 0:
        print("No TP cases found.")
        return

    case = tp_cases.iloc[0]
    complaint_id = case['complaint_id']
    graph_path = Path(f"data/graphs/{complaint_id}.graphml")
    
    if not graph_path.exists():
        print(f"Graph file {graph_path} not found!")
        return

    # 1. Normal forward
    data = build_pyg_data_from_graphml(graph_path, complaint_id, 1.0, case['incident_entity'])
    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    
    with torch.no_grad():
        out_normal_node, out_normal_graph = model(data.x, data.edge_index, batch)
        prob_normal = torch.sigmoid(out_normal_graph).item()

    # 2. Empty edges (only self loops or empty)
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        out_empty_node, out_empty_graph = model(data.x, empty_edge_index, batch)
        prob_empty = torch.sigmoid(out_empty_graph).item()

    # 3. Shuffled edges
    shuffled_edge_index = data.edge_index.clone()
    shuffled_edge_index[1] = shuffled_edge_index[1][torch.randperm(shuffled_edge_index.size(1))]
    with torch.no_grad():
        out_shuffled_node, out_shuffled_graph = model(data.x, shuffled_edge_index, batch)
        prob_shuffled = torch.sigmoid(out_shuffled_graph).item()

    print(f"Complaint: {complaint_id}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Normal Prob: {prob_normal:.4f}")
    print(f"Empty Edges Prob: {prob_empty:.4f}")
    print(f"Shuffled Edges Prob: {prob_shuffled:.4f}")

    # Activation check
    print("\nActivation Check:")
    with torch.no_grad():
        x = data.x
        x1 = model.conv1(x, data.edge_index)
        x1_relu = torch.relu(x1)
        x2 = model.conv2(x1_relu, data.edge_index)
        
        print("Conv1 Output Var:", torch.var(x1, dim=0).mean().item())
        print("Conv2 Output Var:", torch.var(x2, dim=0).mean().item())

if __name__ == "__main__":
    main()
