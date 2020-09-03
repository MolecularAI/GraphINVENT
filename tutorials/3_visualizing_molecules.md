### Visualizing molecules
After generating structures using GraphINVENT, you will probably want to visualize them. We recommend using RDKit for visualizing the molecules as it is has very convenient tools for drawing simple yet elegant grids of molecules.

However, if you generate thousands of molecules, you probably don't want to visualize all of them at once. An example script for visualizing 100 randomly selected molecules from an example SMILES file is shown below:

```
example_visualization_script.py >
import math
import random
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "path/to/file.smi"

# load molecules from file
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1, titleLine=has_header)

n_samples = 100
mols_list = [mol for mol in mols]
mol_to_draw = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

mols_per_row = int(math.sqrt(n_samples))  # determine how many molecules to draw per row

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
labels=list(range(n_samples)  # label structures with a number

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mol_to_draw,
                      molsPerRow=mols_per_row,
                      legends=[str(i) for i in labels],
                      highlightAtomLists=highlight)

img.save(png_filename)
```

Alternatively, you could create a file with 100 randomly sampled molecules, and then visualize the entire file:

```
shuffle -n 100 path/to/file.smi > path/to/file_100_shuffled.smi
```

```
example_visualization_script_2.py >
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "path/to/file_100_shuffled.smi"

# load molecules from file
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1, titleLine=has_header)

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
labels=list(range(n_samples)  # label structures with a number

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mols,
                      molsPerRow=10,
                      legends=[str(i) for i in labels],
                      highlightAtomLists=highlight)

img.save(png_filename)
```