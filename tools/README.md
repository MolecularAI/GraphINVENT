# Tools
This directory contains various tools for analyzing datasets:

* [max_n_nodes.py](./max_n_nodes.py): Gets the maximum number of nodes per molecule in a set of molecules.
* [atom_types.py](./atom_types.py) : Gets the atom types present in a set of molecules.
* [formal_charges.py](./formal_charges.py) : Gets the formal charges present in a set of molecules.
* [combine_HDFs.py](./combine_HDFs.py) : Combines multiple sets of preprocessed HDF files into one set (useful when preprocessing large datasets in batches).

To use any tool in this directory, first activate the GraphINVENT virtual environment, then run:

```
(graphinvent)$ python {script} --smi path/to/file.smi
```

Simply replace *{script}* by the name of the script e.g. *max_n_nodes.py*, and *path/to/file* with the name of the SMILES file to analyze.
