import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runsmcb", action="store_true", default=False, help="run mcb tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runsmcb"):
        # --runsmcb given in cli: do not skip mcb tests
        return
    skip_mcb = pytest.mark.skip(reason="need --runsmcb option to run")
    for item in items:
        if "mcb" in item.keywords:
            item.add_marker(skip_mcb)
