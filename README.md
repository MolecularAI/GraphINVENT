# GraphINVENT

## Description
GraphINVENT is a platform for graph-based molecular generation using graph neural networks. GraphINVENT uses a tiered deep neural network architecture to probabilistically generate new molecules a single bond at a time. All models implemented in GraphINVENT can quickly learn to build molecules resembling training set molecules without any explicit programming of chemical rules. The models have been benchmarked using the MOSES distribution-based metrics, showing how the best GraphINVENT model compares well with state-of-the-art generative models.

## Prerequisites
* Anaconda or Miniconda with Python 3.6 or 3.8.
* CUDA-enabled GPU.

## Tutorials
For detailed guides on how to use GraphINVENT, see the [tutorials](./tutorials/).

## Usage summary

### Setting up the environment
Begin by configuring your environment to have all the proper versions of PyTorch, RDKit, etc installed. To make this more convenient, you can use [./environments/GraphINVENT-env.yml](./environments/GraphINVENT-env.yml), which lists all required packages. The environment can be created from the YAML file by typing:

```
conda env create -f environments/GraphINVENT-env.yml
```

To activate the environment:

```
conda activate GraphINVENT-env
```

### Running the program
GraphINVENT can be run from within the configured environment using:

```
(GraphINVENT-env)$ python main.py
```

This will run a job using the default model and settings specified in [./graphinvent/parameters/defaults.py](graphinvent/parameters/defaults.py). Output will be written to *output/*.

To have a bit more control, one can also specify the output directory using the *--job-dir* flag on the terminal:

```
(GraphINVENT-env)$ python main.py --job-dir "path/to/output/"
```

### Changing the default settings

If a job directory is specified, one can use different settings to the defaults by creating an *input.csv* file in that directory with the desired parameters. An example input file is available at [./output/input.csv](./output/input.csv).

Alternatively, one can modify the default parameters in *graphinvent/parameters/defaults.py*, but this is not recommended. Instead, we have provided a submission script, [submit.py](./submit.py), which can be modified with the desired job parameters and be run as follows:

```
(GraphINVENT-env)$ python submit.py
```

The submission script automatically creates a job directory with an *input.csv* file using the specified parameters and submits the job. Output will be written to the newly created job directory.

### Using other datasets
When using GraphINVENT to train models on new datasets, you must make sure to split the data (using SMILES) into the desired training, testing, and validation sets.

We recommend creating a new directory for each dataset that contains the dataset splits, using the SMILES format, named as follows:
* train.smi
* valid.smi
* test.smi

Running a preprocessing job will create HDF files for each of these splits that contain the processed graph representations for molecular graphs (and subgraphs) in each split.

Furthermore, if different preprocessing parameters will be used for the same dataset, these should be kept in different directories, so as to not overwrite the data. For example, let's say you want to compare a model trained on structures that ignore aromatic bonds with one trained on structures that include aromatic bonds; in this case, the best thing to do is create a two copies of your dataset, e.g. *my_dataset_arom/* and *my_dataset_no_arom/*, and follow the workflow for each copy with the desired parameters.

Some things to keep in mind when using GraphINVENT is that the larger the dataset is, the more disk space will be required to save the processed dataset splits. Furthermore, having additional node features (e.g. atom types) and additional nodes (i.e. greater *max_n_nodes*) in graphs will increase the RAM requirement of jobs, as training data is padded to be the size of the largest graph in the dataset. For reference, the largest dataset we have trained on contains 7 atom types (not counting hydrogen), and molecules with up to 27 heavy atoms; this requires about 10 GB RAM. Larger datasets with a greater variety of node features are thus certainly possible to train on, keeping in mind that if you have very large molecules you might want to cut down the number of node features, and vice versa, as both of these things multiply and lead to an increase in RAM during training/generation. We have not yet examined the edge cases, but estimate that datasets with 10–80 heavy atoms and between 1–15 atom types are in the right range.

### Workflow
The structure if the code is divided into four general workflows:
* preprocessing
* training
* generation
* benchmarking

See the paper for details. To control what type of job is run, simply tune the *job_type* parameter before running a job (possible values include "preprocess", "train", or "generate"); see *Changing the default settings* above.

To start building molecules right away, an example dataset (see *Examples* below) has been provided, along with a trained model.

During preprocessing jobs, or `job_type="preprocess"`, the following will be written to the specified *dataset_dir*:
* 3 HDF files (*train.h5*, *valid.h5*, and *test.smi*)
* *preprocessing_params.csv*, containing parameters used in preprocessing the dataset
* *train.csv*, containing training set properties (e.g. histograms of number of nodes per molecule, number of edges per node, etc)

During training jobs, or `job_type="train"`, the following will be written to the specified *job_dir* and updated as the model trains:
* *generation.csv*, containing various evaluation metrics for the generated set, calculated during evaluation epochs
* *convergence.csv*, containing the loss and learning rate for every epoch
* *validation.csv*, containing model scores (e.g. NLLs, UC-JSD), calculated during evaluation epochs
* *model_restart_{epoch}.pth*, which are the model states for use in restarting jobs, or running generation/validation jobs with a trained model
* *generation/*, a directory containing structures generated during evaluation epochs (\*.smi), as well as information on each structure's NLL (\*.nll) and validity (\*.valid)

During generation jobs, or `job_type="generate"`, the following will be written to the *generation/* directory within the specified *job_dir*:
* *epochGEN{epoch}_{batch}.smi*, containing molecules generated at the epoch specified by *generation_epoch*
* *epochGEN{epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.csv* file will be updated with the various evaluation metrics for the generated structures.

During validation jobs, or `job_type="test"`, the *generation.csv* file will be updated with the evaluation metrics for the test set structures.

For benchmarking jobs, see below.


### Benchmarking models
Models can be easily benchmarked using MOSES. To do this, we recommend reading the MOSES documentation, available at https://github.com/molecularsets/moses. If you want to compare to previously benchmarked models you will have to train models using the MOSES datasets.

Once you have a satisfactorily trained model, you can run a generation job to create 30,000 new structures. GraphINVENT will write these molecules to a SMILES file, which will be created in the *generation/* dir within the job directory. The generated structures can then be used as the <generated dataset> in MOSES evaluation jobs.

## Examples
An example training set is available in [./data/gdb13_1K/](./data/gdb13_1K/). It is a small (1K) subset of GDB-13 and is already preprocessed.

## Hyperparameters

### Default parameters and hyperparameters
The parameters and hyperparameters in *./graphinvent/parameters/defaults.py* are the defaults that will be used for any job unless they are overwritten by an *input.csv* in the job directory. The hyperparameters in *./graphinvent/parameters/defaults.py* are generally "good" for these models, but of course the "ideal" hyperparameters will vary slightly depending on the dataset. In particular, parameters related to the learning rate decay are sensitive to the dataset, so a bit of experimentation here is recommended when changing datasets as these parameters can make a difference between an "okay" model and a well-trained model. These parameters are:

* *init_lr*
* *min_rel_lr*
* *lrdf*
* *lrdi*

Note that not all parameters defined in *./graphinvent/parameters/defaults.py* are model-related hyperparameters; many are simply practical parameters and settings, such as the path to the datasets being studied.

### Dataset parameters
The most important parameters to overwrite are those related to the dataset in question, such as the atom types present in the dataset, the formal charges present in the dataset, etc (unless you happen to be studying subsets of GDB-13, in which case you won't need to change the defaults). These can be determined using popular cheminformatics toolkits (such as RDKit or OEChem).

You will also need to know the atom count for the largest molecule in your training set. GraphINVENT does not calculate this as it is slow for very large datasets, so it is better if this is calculated beforehand. We have provided a script, [./tools/get_max_n_nodes.py](./tools/get_max_n_nodes.py) that can do this for you.

## Contributors
[@rociomer](https://www.github.com/rociomer)

[@rastemo](https://www.github.com/rastemo)

[@edvardlindelof](https://www.github.com/edvardlindelof)

## Contributions

Contributions are welcome in the form of issues or pull requests. To report a bug, please submit an issue.

## References
If you use GraphINVENT in your research, please reference our [publication](https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137/1):

```
@article{mercado2020graph,
  author = "Rocío Mercado and Tobias Rastemo and Edvard Lindelöf and Günter Klambauer and Ola Engkvist and Hongming Chen and Esben Jannik Bjerrum",
  title = "{Graph Networks for Molecular Design}",
  year = "2020",
  journal = "ChemRxiv preprint doi:10.26434/chemrxiv.12843137.v1"
  url = "https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137",
  doi = "10.26434/chemrxiv.12843137.v1"
}
```

Additional details related to the development of GraphINVENT are available in our [technical note](https://chemrxiv.org/articles/preprint/Practical_Notes_on_Building_Molecular_Graph_Generative_Models/12888383/1). You might find this note useful if you're interested in either exploring different hyperparameters or developing your own generative models. The reference to that document is:

```
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
The MPNN implementations used in this work were pulled from Edvard Lindelöf's repo in October 2018, while he was a masters student in the MAI group. This work is available at:

https://github.com/edvardlindelof/graph-neural-networks-for-drug-discovery.

His master's thesis, describing the EMN implementation, can be found here:

https://odr.chalmers.se/handle/20.500.12380/256629.

#### MOSES
The MOSES repo is available at https://github.com/molecularsets/moses.

#### GDB-13
The example dataset provided is a subset of GDB-13. This was obtained by randomly sampling 1000 structures from the entire GDB-13 dataset. The full dataset is available for download at http://gdb.unibe.ch/downloads/.


## License

GraphINVENT is licensed under the MIT license and is free and provided as-is.

## Link
https://github.com/MolecularAI/GraphINVENT/
