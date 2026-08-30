import os

path = 'src/graphsage_classifier.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

def_idx = c.find('def evaluate_test_set')
if def_idx != -1:
    old_func = c[def_idx:]
    new_func = old_func.replace('for batch in val_loader:', 'for batch in test_loader:')
    c = c[:def_idx] + new_func

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
