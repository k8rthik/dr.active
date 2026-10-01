import pytest
import torch

from dractive.config import GNNConfig
from dractive.gnn.graph import collate_graphs, mol_to_graph
from dractive.gnn.model import AffinityGNN, resolve_device, segment_mean, segment_sum

SMALL = GNNConfig(hidden_dim=16, num_layers=2, target_embedding_dim=4)


def _batch(pairs):
    return collate_graphs([mol_to_graph(s, t, y) for s, t, y in pairs])


def test_segment_sum_groups_rows():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    ids = torch.tensor([0, 0, 1])
    assert segment_sum(values, ids, 2).squeeze(-1).tolist() == [3.0, 3.0]


def test_segment_mean_groups_rows():
    values = torch.tensor([[1.0], [3.0], [10.0]])
    ids = torch.tensor([0, 0, 1])
    assert segment_mean(values, ids, 2).squeeze(-1).tolist() == [2.0, 10.0]


def test_segment_mean_handles_empty_group():
    values = torch.tensor([[4.0]])
    ids = torch.tensor([1])
    out = segment_mean(values, ids, 2).squeeze(-1)
    assert out.tolist() == [0.0, 4.0]


def test_forward_returns_one_prediction_per_graph():
    model = AffinityGNN(SMALL)
    batch = _batch([("CCO", "EGFR", 6.0), ("c1ccccc1O", "HERG", 5.0)])
    out = model(batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic_in_eval_mode():
    model = AffinityGNN(SMALL).eval()
    batch = _batch([("CCO", "EGFR", 6.0)])
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch))


def test_prediction_depends_on_target_embedding():
    model = AffinityGNN(SMALL).eval()
    with torch.no_grad():
        a = model(_batch([("CCO", "EGFR", 0.0)]))
        b = model(_batch([("CCO", "ACHE", 0.0)]))
    assert not torch.allclose(a, b)


def test_batching_matches_single_graph_predictions():
    model = AffinityGNN(SMALL).eval()
    pairs = [("CCO", "EGFR", 6.0), ("c1ccncc1", "DRD2", 7.0)]
    with torch.no_grad():
        batched = model(_batch(pairs))
        singles = torch.cat([model(_batch([p])) for p in pairs])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_molecule_with_no_bonds_does_not_produce_nan():
    model = AffinityGNN(SMALL).eval()
    with torch.no_grad():
        out = model(_batch([("[Na+]", "EGFR", 5.0)]))
    assert torch.isfinite(out).all()


def test_gradients_flow_to_all_parameters():
    model = AffinityGNN(SMALL)
    batch = _batch([("CCO", "EGFR", 6.0), ("c1ccccc1", "JAK2", 7.0)])
    loss = torch.nn.functional.mse_loss(model(batch), batch.labels)
    loss.backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    ]
    # only the embeddings of targets absent from this batch may be untouched
    assert all("target_embedding" in name for name in unused), unused


def test_parameter_count_is_reported():
    assert AffinityGNN(SMALL).num_parameters() > 0


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        AffinityGNN(GNNConfig(num_layers=0))
    with pytest.raises(ValueError):
        AffinityGNN(GNNConfig(hidden_dim=0))


@pytest.mark.parametrize("requested", ["cpu", "auto"])
def test_resolve_device_returns_usable_device(requested):
    device = resolve_device(requested)
    assert isinstance(device, torch.device)
    torch.zeros(1, device=device)


def test_resolve_device_rejects_nonsense():
    with pytest.raises(ValueError):
        resolve_device("quantum")
