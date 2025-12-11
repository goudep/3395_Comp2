"""
Fix the milestone2_ultimate.ipynb notebook by replacing the validate() call
with predict() for test data inference.
"""

import json

notebook_path = "milestone2_ultimate.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix the problematic cell
fixed = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for i, line in enumerate(source):
            # Replace the problematic validate call for test data
            if 'validate(model, test_loader_orig' in line:
                # Replace with predict call
                new_line = line.replace(
                    '_, _, probs1 = validate(model, test_loader_orig, criterion, device)',
                    'probs1 = predict(model, test_loader_orig, device)'
                )
                new_source.append(new_line)
                fixed = True
                print(f"Fixed line: {line.strip()}")
                print(f"     -> {new_line.strip()}")
            else:
                new_source.append(line)
        cell['source'] = new_source

# Clear all outputs
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# Write the fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4, ensure_ascii=False)

if fixed:
    print("\n✅ Notebook fixed successfully!")
    print("Please re-run the notebook from the beginning.")
else:
    print("\n⚠️ Could not find the problematic line. Manual fix may be needed.")
    print("Replace this line:")
    print("  _, _, probs1 = validate(model, test_loader_orig, criterion, device)")
    print("With:")
    print("  probs1 = predict(model, test_loader_orig, device)")
