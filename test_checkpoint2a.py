import sys
from pathlib import Path
import torch
from torch_geometric.data import Data
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.graphsage_classifier import DualHeadGraphSAGE, MODELS_DIR

def run_test():
    model = DualHeadGraphSAGE(input_dim=13, hidden_dim=64, dropout=0.2)
    state_dict = torch.load(MODELS_DIR / "graphsage_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy zero tensor for features
    x = torch.zeros((1, 13), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    batch = torch.zeros(1, dtype=torch.long)
    
    with torch.no_grad():
        out_node, out_graph, emb = model(x, edge_index, batch)
        prob = torch.sigmoid(out_graph).item()
        
    print(f"Zero vector probability: {prob}")
    
if __name__ == "__main__":
    run_test()
