import urllib.request
import json
url = 'http://localhost:8000/api/entities/locations'
req = urllib.request.Request(url)
response = urllib.request.urlopen(req)
data = json.loads(response.read().decode('utf-8'))
found = 0
for item in data:
    if 'Unknown City' in item.get('city', ''):
        print(f"Entity: {item['entity_id']}, City: '{item['city']}', State: '{item['state']}', Type: {item['entity_type']}")
        found += 1
        if found >= 5: break
