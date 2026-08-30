import os

path = 'src/graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'train_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] not in test_set and getattr(d, "complaint_id", [""])[0] not in val_set]',
    'train_raw = [d for d in raw_dataset if getattr(d, "complaint_id", "") not in test_set and getattr(d, "complaint_id", "") not in val_set]'
)
c = c.replace(
    'val_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] in val_set]',
    'val_raw = [d for d in raw_dataset if getattr(d, "complaint_id", "") in val_set]'
)
c = c.replace(
    'test_raw = [d for d in raw_dataset if getattr(d, "complaint_id", [""])[0] in test_set]',
    'test_raw = [d for d in raw_dataset if getattr(d, "complaint_id", "") in test_set]'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
