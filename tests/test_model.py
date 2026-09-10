import torch
from torch_geometric.data import Batch, HeteroData

from bapred2.config import ModelConfig
from bapred2.model import model_from_sample


def make_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["protein"].x = torch.randn(5, 37, generator=g)
    data["protein"].pos_enc = torch.randn(5, 20, generator=g)
    data["ligand"].x = torch.randn(3, 37, generator=g)
    data["ligand"].pos_enc = torch.randn(3, 20, generator=g)
    p_rel = ("protein", "intra", "protein")
    l_rel = ("ligand", "intra", "ligand")
    c_rel = ("protein", "contact", "ligand")
    data[p_rel].edge_index = torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]])
    data[p_rel].edge_attr = torch.randn(8, 25, generator=g)
    data[l_rel].edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    data[l_rel].edge_attr = torch.randn(4, 25, generator=g)
    data[c_rel].edge_index = torch.tensor([[0,1,2,3,4],[0,0,1,2,2]])
    data[c_rel].edge_attr = torch.randn(5, 31, generator=g)
    data.y = torch.tensor([6.5])
    return data


def test_forward_and_recycle_sweep():
    sample = make_graph()
    model = model_from_sample(sample, ModelConfig(hidden_dim=64, prelude_layers=1))
    batch = Batch.from_data_list([sample, make_graph(1)])
    for r in [1, 2, 4, 8]:
        pred, aux = model(batch, recycles=r, return_aux=True)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        assert aux["cycle_delta"].shape == (r,)


def test_backward():
    sample = make_graph()
    model = model_from_sample(sample, ModelConfig(hidden_dim=32, prelude_layers=1))
    batch = Batch.from_data_list([sample])
    loss = (model(batch, recycles=3) - batch.y.reshape(-1)).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
