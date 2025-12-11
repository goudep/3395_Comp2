import json
import os

nb_path = r"c:\Users\yudim\Downloads\IFT3395_Competition2\Milestone2.ipynb"

print(f"Reading notebook from: {nb_path}")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the robust installation cell
install_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Installing tqdm in the current kernel environment\n",
        "import sys\n",
        "!{sys.executable} -m pip install tqdm\n"
    ]
}

# Check if we already have it
if len(nb['cells']) > 0:
    first_source = nb['cells'][0].get('source', [])
    # If the first cell is markdown, check the second
    if len(nb['cells']) > 1 and nb['cells'][0]['cell_type'] == 'markdown':
        second_source = nb['cells'][1].get('source', [])
        if second_source and "!{sys.executable}" in second_source[-1]:
             print("Robust installation cell already present at index 1.")
             # We can force update it just in case
             nb['cells'][1] = install_cell
        else:
             print("Inserting installation cell at index 1.")
             nb['cells'].insert(1, install_cell)
    else:
        # If first cell is not markdown (weird but possible), check if it's the install cell
        if first_source and "!{sys.executable}" in first_source[-1]:
             print("Robust installation cell already present at index 0.")
             nb['cells'][0] = install_cell
        else:
             print("Inserting installation cell at index 0.")
             nb['cells'].insert(0, install_cell)

print(f"Writing notebook back to: {nb_path}")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
print("Done.")
