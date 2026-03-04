#!/usr/bin/env python3
"""Search OpenNeuro for pain-related EEG datasets."""
import json
import urllib.request

all_datasets = []
cursor = None
for page in range(20):
    q = '{ datasets(first: 100, modality: "eeg"'
    if cursor:
        q += ', after: "' + cursor + '"'
    q += ') { edges { cursor node { id name } } pageInfo { hasNextPage endCursor } } }'
    
    req = urllib.request.Request(
        'https://openneuro.org/crn/graphql',
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    
    edges = data.get("data", {}).get("datasets", {}).get("edges", [])
    if not edges:
        break
    all_datasets.extend(edges)
    
    pi = data["data"]["datasets"]["pageInfo"]
    if not pi["hasNextPage"]:
        break
    cursor = pi["endCursor"]

print(f"Total EEG datasets on OpenNeuro: {len(all_datasets)}")

kw = [
    "pain", "nocicep", "laser", "noxious", "thermal", "heat",
    "somato", "evoked potential", "nociception", "LEP ",
    "spinal cord", "nociceptive",
]
print("\n--- Pain/nociception related ---")
for edge in all_datasets:
    n = edge["node"]
    name = (n.get("name", "") or "").lower()
    if any(k.lower() in name for k in kw):
        print(f"  {n['id']}: {n['name']}")
