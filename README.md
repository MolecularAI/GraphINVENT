# GraphINVENT

![cover image](./cover-image.png)

## Description
GraphINVENT is a platform for graph-based molecular generation using graph neural networks. GraphINVENT uses a tiered deep neural network architecture to probabilistically generate new molecules a single bond at a time. All models implemented in GraphINVENT can quickly learn to build molecules resembling training set molecules without any explicit programming of chemical rules. The models have been benchmarked using the MOSES distribution-based metrics, showing how the best GraphINVENT model compares well with state-of-the-art generative models.

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* CUDA-enabled GPU.

## Tutorials
For detailed guides on how to use GraphINVENT, see the [tutorials](./tutorials/).

## Examples
An example training set is available in [./data/gdb13_1K/](./data/gdb13_1K/). It is a small (1K) subset of GDB-13 and is already preprocessed.

## Contributors
[@rociomer](https://www.github.com/rociomer)

[@rastemo](https://www.github.com/rastemo)

[@edvardlindelof](https://www.github.com/edvardlindelof)

## Contributions

Contributions are welcome in the form of issues or pull requests. To report a bug, please submit an issue.

## References
### Relevant publications
If you use GraphINVENT in your research, please reference our [publication](https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137/1).

Additional details related to the development of GraphINVENT are available in our [technical note](https://chemrxiv.org/articles/preprint/Practical_Notes_on_Building_Molecular_Graph_Generative_Models/12888383/1). You might find this note useful if you're interested in either exploring different hyperparameters or developing your own generative models.

The references in BibTex format are available below:

```
@article{mercado2020graph,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Graph Networks for Molecular Design}",
  year = "2020",
  journal = "ChemRxiv preprint doi:10.26434/chemrxiv.12843137.v1"
  url = "https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137",
  doi = "10.26434/chemrxiv.12843137.v1"
}

@article{mercado2020practical,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Practical Notes on Building Molecular Graph Generative Models}",
  year = "2020",
  journal = "ChemRxiv preprint doi:10.26434/chemrxiv.12888383.v1"
  url = "https://chemrxiv.org/articles/preprint/Practical_Notes_on_Building_Molecular_Graph_Generative_Models/12888383",
  doi = "10.26434/chemrxiv.12888383.v1"
}
```

### Related work
#### MPNNs
The MPNN implementations used in this work were pulled from Edvard Lindelöf's repo in October 2018, while he was a masters student in the MAI group. This work is available at

https://github.com/edvardlindelof/graph-neural-networks-for-drug-discovery.

His master's thesis, describing the EMN implementation, can be found at

https://odr.chalmers.se/handle/20.500.12380/256629.

#### MOSES
The MOSES repo is available at https://github.com/molecularsets/moses.

#### GDB-13
The example dataset provided is a subset of GDB-13. This was obtained by randomly sampling 1000 structures from the entire GDB-13 dataset. The full dataset is available for download at http://gdb.unibe.ch/downloads/.


## License

GraphINVENT is licensed under the MIT license and is free and provided as-is.

## Link
https://github.com/MolecularAI/GraphINVENT/
