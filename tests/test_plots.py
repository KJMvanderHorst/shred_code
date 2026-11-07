import matplotlib

matplotlib.use("Agg")

from get_shredded.plots import method_color, plot_exact_k


def test_plot_exact_k_mean_over_seeds():
    results = {
        ("shred__missing", s): {"metrics": {"clean": {"err": 0.0},
                                            **{f"missing_oracle_k{k}": {"err": k + s} for k in range(3)},
                                            "missing_oracle_p0.3": {"err": 9.0}}}
        for s in (0, 2)
    }
    ax = plot_exact_k(results, ["shred__missing", "absent__run"], "missing_oracle")
    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == [1.0, 2.0, 3.0]


def test_garbage_run_shares_method_colour():
    assert method_color("shred__garbage") == method_color("shred__missing")
