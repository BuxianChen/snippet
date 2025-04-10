import requests

resp = requests.post("http://localhost:8888/set", json={"key": "test", "value": "a"}).json()
print(resp)