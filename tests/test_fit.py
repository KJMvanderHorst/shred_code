import torch

from get_shredded.model import SDN, TimeSeriesDataset, fit


def test_fit_without_early_stopping_runs_all_epochs_and_keeps_final_weights() -> None:
    torch.manual_seed(0)
    x, y = torch.rand(16, 4), torch.rand(16, 6)
    valid = TimeSeriesDataset(torch.rand(8, 4), torch.rand(8, 6))  # unrelated targets: validation won't improve
    model = SDN(4, 6, l1=8, l2=8)
    history = fit(model, TimeSeriesDataset(x, y), valid, num_epochs=200, patience=1, early_stopping=False)
    assert len(history) == 11  # epoch 1 + every 20 epochs
    model.eval()
    with torch.no_grad():
        final_val = torch.linalg.norm(model(valid.X) - valid.Y) / torch.linalg.norm(valid.Y)
    assert torch.isclose(final_val, history[-1])
