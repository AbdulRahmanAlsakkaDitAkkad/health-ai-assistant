import pytest
from fastapi.testclient import TestClient

from assistant.service.api import app

@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
