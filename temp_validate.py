import json
f = open('heston.ipynb', 'r')
nb = json.load(f)
f.close()
code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f"Valid notebook: {len(nb['cells'])} cells total ({code} code, {md} markdown)")
