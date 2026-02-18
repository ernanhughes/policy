import pytest

@pytest.fixture
def dummy_context():
    return {"test": True}
