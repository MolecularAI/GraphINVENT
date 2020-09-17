## Visualizing molecules
After generating structures using GraphINVENT, you will almost certainly want to visualize them. Below we provide some examples using RDKit for visualizing the molecules in simple but elegant grids.

### Drawing a grid of molecules
Assuming you use the trained models to generate thousands (if not more) molecules, you *probably* don't want to visualize all of them in one massive grid. A more reasonable thing to do is to randomly sample a small subset for visualization.

An example script for drawing 100 randomly selected molecules is shown below:

```
example_visualization_script.py >
import math
import random
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "path/to/file.smi"

# load molecules from file
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1)

n_samples = 100
mols_list = [mol for mol in mols]
mols_sampled = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

mols_per_row = int(math.sqrt(n_samples))            # make a square grid

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
labels=list(range(n_samples))       # label structures with a number

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mols_sampled,
                      molsPerRow=mols_per_row,
                      legends=[str(i) for i in labels])

img.save(png_filename)
```

Alternatively, you could first randomly sample 100 molecules from your source file, save them in a new file, and draw everything in the new file:

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
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1)

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
labels=list(range(n_samples))       # label structures with a number

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mols,
                      molsPerRow=10,
                      legends=[str(i) for i in labels])

img.save(png_filename)
```

### Filtering out invalid entries
By default, GraphINVENT writes a "Xe" placeholder when an invalid molecular graph is generated, as an invalid molecular graph cannot be converted to a SMILES string for saving. The placeholder is used because the NLL is written for all generated graphs in a separate file, where the same line number in the \*.nll file corresponds to the same line number in the \*.smi file. Similarly, if an empty graph samples an invalid action as the first action, then no SMILES can be generated for an empty graph, so the corresponding line for an empty graph in a SMILES file contains only the "ID" of the molecule.

For visualization, you might be interested in viewing only the valid molecular graphs. The SMILES for the generated molecules can thus be post-processed as follows to remove empty and invalid entries from a file before visualization:

```
sed -i "/Xe/d" path/to/file.smi          # remove "Xe" placeholders from file
sed -i "/^ [0-9]\+$/d" path/to/file.smi  # remove empty graphs from file
```
