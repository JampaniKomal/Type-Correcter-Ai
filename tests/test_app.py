"""Tests for the Flask web app, via the test client (no running server)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app as flask_app  # noqa: E402


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Type Correcter Ai" in resp.data


def test_predict_corrects_text(client):
    resp = client.post("/predict", json={"text": "teh recieve is beleive"})
    assert resp.status_code == 200
    assert resp.get_json()["correction"] == "the receive is believe"


def test_predict_requires_text(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_handles_empty_string(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    assert resp.get_json()["correction"] == ""
