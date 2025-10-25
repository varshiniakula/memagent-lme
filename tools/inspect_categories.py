import json, collections
from pathlib import Path

p = Path("data/longmemeval.json")
data = json.loads(p.read_text(encoding="utf-8"))
if isinstance(data, dict): data = data.get("data") or data

num_convs = len(data)
num_qas = sum(len(c.get("qa", [])) for c in data)
hist = collections.Counter(q.get("category") for c in data for q in c.get("qa", []))

print("Conversations:", num_convs)
print("Total QAs:", num_qas)
print("Category histogram:", dict(hist))
print("First 5 conv IDs:", [c.get("id") for c in data[:5]])
