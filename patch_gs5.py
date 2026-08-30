import os

path = 'src/graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'pos_weight = float(train_neg / train_pos) if train_pos > 0 else 1.0',
    '''train_neg = sum(1 for d in train_raw if d.y.item() == 0.0)
    train_pos = sum(1 for d in train_raw if d.y.item() == 1.0)
    pos_weight = float(train_neg / train_pos) if train_pos > 0 else 1.0'''
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
