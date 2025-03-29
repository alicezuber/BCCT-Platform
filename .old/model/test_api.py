import pytest
from api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_add(client):
    response = client.post('/calculate', json={'operation': '+', 'num1': 10, 'num2': 5})
    assert response.status_code == 200
    assert response.json['result'] == 15

def test_subtract(client):
    response = client.post('/calculate', json={'operation': '-', 'num1': 10, 'num2': 5})
    assert response.status_code == 200
    assert response.json['result'] == 5

def test_multiply(client):
    response = client.post('/calculate', json={'operation': '*', 'num1': 10, 'num2': 5})
    assert response.status_code == 200
    assert response.json['result'] == 50

def test_divide(client):
    response = client.post('/calculate', json={'operation': '/', 'num1': 10, 'num2': 5})
    assert response.status_code == 200
    assert response.json['result'] == 2

def test_divide_by_zero(client):
    response = client.post('/calculate', json={'operation': '/', 'num1': 10, 'num2': 0})
    assert response.status_code == 400
    assert response.json['error'] == 'Cannot divide by zero'

def test_invalid_operation(client):
    response = client.post('/calculate', json={'operation': '^', 'num1': 10, 'num2': 5})
    assert response.status_code == 400
    assert response.json['error'] == 'Invalid operation'