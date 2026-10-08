import pytest

from triggercollide.demo import write_demo_library


@pytest.fixture()
def demo_dir(tmp_path):
    return write_demo_library(tmp_path / "loras")
