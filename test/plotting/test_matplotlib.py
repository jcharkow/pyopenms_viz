"""
test/plotting/test_matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
import pandas as pd
from pyopenms_viz.testing import MatplotlibSnapshotExtension


@pytest.fixture
def snapshot_mpl(snapshot):

    return snapshot.use_extension(MatplotlibSnapshotExtension)


@pytest.fixture
def raw_data():
    return pd.read_csv("../test_data/ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture
def annotation_data():
    return pd.read_csv("../test_data/ionMobilityTestChromatogramDf.tsv", sep="\t")


@pytest.fixture(scope="session", autouse=True)
def load_backend():
    import pandas as pd

    pd.set_option("plotting.backend", "ms_matplotlib")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(by="Annotation"),
        dict(scale_intensity=True),
        dict(by="Annotation", scale_intensity=True),
        dict(xlabel="RETENTION", ylabel="INTENSITY"),
        dict(grid=True),
    ],
)
def test_chromatogram_plot_mpl(raw_data, snapshot_mpl, kwargs):
    out = raw_data.plot(x="rt", y="int", kind="chromatogram", show_plot=False, **kwargs)

    assert snapshot_mpl == out.superFig
