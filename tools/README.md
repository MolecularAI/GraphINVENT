# Tools
This directory contains various tools for analyzing datasets:

* [max_n_nodes.py](./max_n_nodes.py): Gets the maximum number of nodes per molecule in a set of molecules.
* [atom_types.py](./atom_types.py) : Gets the atom types present in a set of molecules.
* [formal_charges.py](./formal_charges.py) : Gets the formal charges present in a set of molecules.
* [tdc-create-dataset.py](./tdc-create-dataset.py) : Downloads a dataset, such as ChEMBL or MOSES, from the Therapeutics Data Commons (TDC).
* [submit-split-preprocessing-supercloud.py](./submit-split-preprocessing-supercloud.py) : Example submission script for preprocessing a very large dataset in parallel.

---

To use the first 3 tools in this directory ([max_n_nodes.py](./max_n_nodes.py), [atom_types.py](./atom_types.py), or [formal_charges.py](./formal_charges.py)), first activate the GraphINVENT virtual environment, then run:

```
(graphinvent)$ python {script} --smi path/to/file.smi
```

Simply replace *{script}* by the name of the script e.g. *max_n_nodes.py*, and *path/to/file* with the name of the SMILES file to analyze.

---
If you would like to download a dataset such as ChEMBL or MOSES from the TDC and preprocess it slightly (e.g. remove molecular with high formal charges, filter to molecules with <= 80 heavy atoms, etc), then you can use the [tdc-create-dataset.py](./tdc-create-dataset.py) script.

To use script to download, for example, the MOSES dataset, run (from within the GraphINVENT environment):
```
(graphinvent)$ python tdc-create-dataset.py --dataset MOSES
```

You can change the flag to speficy other datasets available via the TDC.

Furthermore, you can manually edit the script to do other things you would like (for instance, set the number of heavy atoms and formal charges to filter).

---

In some cases, if you have a really large dataset, it might be easier to preprocess it in pieces (i.e. in parallel on different nodes) rather than all in serial. To do this, you can use the [submit-split-preprocessing-supercloud.py](./submit-split-preprocessing-supercloud.py) script. 

To use it, you will first need to split your dataset by running, **from within an interactive session**, the following command:
```
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type split
```

Then, once the dataset has been split, you can submit the separate splits as individual preprocessing jobs as follows:
```
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type submit
```

When the above jobs have completed, you can aggregate the generated HDFs for each dataset split into a single HDF in the main dataset dir:
```
(graphinvent)$ python submit-split-preprocessing-supercloud.py --type aggregate
```