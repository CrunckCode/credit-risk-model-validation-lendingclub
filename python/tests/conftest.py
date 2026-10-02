import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # lets tests import fixtures.make_fixture

from credit_validation import config, data, targets, features, splits  # noqa: E402
from fixtures import make_fixture  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rawdata")
    make_fixture.write_fixture_csv(d / config.RAW_FILE)
    make_fixture.make_defaults_csv(make_fixture.make_raw_frame(), d / config.DEFAULTS_FILE)
    return d


@pytest.fixture(scope="session")
def loans(raw_dir):
    df = data._clean(data._read_raw(raw_dir / config.RAW_FILE, None))
    return features.build_features(targets.add_targets(df))


@pytest.fixture(scope="session")
def fixture_splits(loans):
    return splits.time_splits(loans)


def pytest_collection_modifyitems(config, items):
    from credit_validation import config as cfg
    if not (cfg.RAW_DIR / cfg.RAW_FILE).exists():
        skip = pytest.mark.skip(reason="real raw data not available")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
