import json
import os

nb_path = r"c:\Users\yudim\Downloads\IFT3395_Competition2\Milestone2.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create a new code cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install tqdm\n"
    ]
}

# Check if the cell already exists to avoid duplicates
first_cell_source = nb['cells'][0]['source']
if not (first_cell_source and "!pip install tqdm" in first_cell_source[0]):
   # Insert at the beginning (index 0 for markdown title, index 1 for code?) 
   # Actually it's cleaner to put it as the very first cell or just before imports.
   # The notebook starts with a markdown cell (title). Let's put it after that, at index 1.
   nb['cells'].insert(1, new_cell)
   
   with open(nb_path, 'w', encoding='utf-8') as f:
       json.dump(nb, f, indent=4)
   print("Successfully added installation cell.")
else:
   print("Installation cell already exists.")
