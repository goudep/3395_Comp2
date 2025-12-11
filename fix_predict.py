"""
Fix milestone2_ultimate.ipynb by adding the missing `predict` function.

Run: python fix_predict.py
"""
import json

notebook_path = "milestone2_ultimate.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with helper functions (contains 'def validate')
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def validate' in source and 'def predict' not in source:
            print(f"Found helper functions cell at index {i}")
            
            # Add the predict function after validate
            predict_code = [
                "\n",
                "\n",
                "@torch.no_grad()\n",
                "def predict(model, loader, device):\n",
                '    """Predict on test data (no labels)."""\n',
                "    model.eval()\n",
                "    probs = []\n",
                "    for imgs in loader:\n",
                "        imgs = imgs.to(device)\n",
                "        outputs = model(imgs)\n",
                "        probs.append(F.softmax(outputs, dim=1).cpu().numpy())\n",
                "    return np.concatenate(probs)"
            ]
            
            # Append to the cell source
            cell['source'].extend(predict_code)
            print("Added `predict` function to the cell.")
            break
else:
    print("Could not find the helper functions cell or `predict` already exists.")

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print(f"\nNotebook '{notebook_path}' has been updated!")
print("Please restart your kernel and re-run all cells.")
