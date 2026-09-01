"""Minimal API test. /health is hermetic; /ask needs a built index + Ollama,
so it's not covered here (integration test to add once the index exists)."""

import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "app"))
from main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
