import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from lib.plotting import _save_and_maybe_show


def test_save_and_maybe_show_forces_pdf_and_strips_titles(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("Diagnostic title")
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")

    _save_and_maybe_show(
        fig,
        "diagnostic.png",
        directory=tmp_path,
        file_format="png",
        pad_inches=0.0,
    )

    assert (tmp_path / "diagnostic.pdf").exists()
    assert not (tmp_path / "diagnostic.png").exists()
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "x label"
    assert ax.get_ylabel() == "y label"
    assert ax.get_xlim() == (0.0, 1.0)
