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


def test_cosine_schedule_decays_lr_to_zero() -> None:
    torch.manual_seed(0)
    ds = TimeSeriesDataset(torch.rand(16, 4), torch.rand(16, 6))
    model = SDN(4, 6, l1=8, l2=8)
    seen = []
    real_adam = torch.optim.Adam

    def spy(*args, **kwargs):
        opt = real_adam(*args, **kwargs)
        seen.append(opt)
        return opt

    torch.optim.Adam = spy
    try:
        fit(model, ds, ds, num_epochs=40, lr=1e-2, early_stopping=False, lr_schedule="cosine")
    finally:
        torch.optim.Adam = real_adam
    assert seen[0].param_groups[0]["lr"] < 1e-8
    try:
        fit(model, ds, ds, num_epochs=1, lr_schedule="step")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lr_schedule accepted")
