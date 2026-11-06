import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from ilos.api import app

client = TestClient(app)
SCENARIO = {'width': 3, 'height': 2, 'vehicles': [{'id': 'A', 'start': [0, 0], 'goal': [2, 0]}]}

def test_plan_and_health(monkeypatch):
    monkeypatch.delenv('ILOS_API_KEY', raising=False)
    assert client.get('/health').json()['status'] == 'ok'
    res = client.post('/plan', json={'scenario': SCENARIO})
    assert res.status_code == 200
    assert res.json()['paths']['A'][-1] == [2, 0]

def test_invalid_and_impossible_scenarios(monkeypatch):
    monkeypatch.delenv('ILOS_API_KEY', raising=False)
    assert client.post('/plan', json={'scenario': {**SCENARIO, 'width': True}}).status_code == 422
    assert client.post('/plan', json={'scenario': {**SCENARIO, 'obstacles': [[2, 0]]}}).status_code == 422
    assert client.post('/plan', json={'scenario': SCENARIO, 'horizon': 0}).status_code == 422

def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv('ILOS_API_KEY', 'test-only-key')
    assert client.post('/plan', json={'scenario': SCENARIO}).status_code == 401
    assert client.post('/plan', json={'scenario': SCENARIO}, headers={'X-API-Key': 'test-only-key'}).status_code == 200

def test_model_paths_cannot_be_supplied_by_clients(monkeypatch):
    monkeypatch.delenv('ILOS_API_KEY', raising=False)
    monkeypatch.delenv('ILOS_PPO_MODEL', raising=False)
    assert client.post('/plan', json={'scenario': SCENARIO, 'backend': 'ppo'}).status_code == 503
    assert client.post('/plan', json={'scenario': SCENARIO, 'model': '/tmp/unknown.zip'}).status_code == 422
