import pytest
from fastapi.testclient import TestClient

app_module = pytest.importorskip("app.main")
app = app_module.app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'


def test_invalid_name_guardrail():
    with TestClient(app) as client:
        response = client.post('/screening/run', json={'full_name': '1'})
        assert response.status_code in (400, 422)
