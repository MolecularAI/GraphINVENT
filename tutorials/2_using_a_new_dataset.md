## Using a new dataset in GraphINVENT
In this tutorial, you will be guided through the steps of using a new dataset in GraphINVENT.

### Selecting a new dataset
Before getting carried away with the possibilities of molecular graph generative models, it should be clear that the GraphINVENT models are computationally demanding, especially compared to string-based models. As such, you should keep in mind the capabilities of your system when selecting a new dataset to study, such as how much disk space you have available, how much RAM, and how fast is your GPU.

In our recent [publication](https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137/1), we report the computational requirements for Preprocessing, Training, Generation, and Benchmarking jobs using the various GraphINVENT models. We summarize some of the results here for the largest dataset we trained on:

| Dataset                                                          | Train | Test | Valid | Largest Molecule | Atom Types              | Formal Charges |
|---|---|---|---|---|---|---|
| [MOSES](https://github.com/molecularsets/moses/tree/master/data) | 1.5M  | 176K | 10K   | 27 atoms         | {C, N, O, F, S, Cl, Br} | {0}            |

The disk space used by the different splits, before and after preprocessing (using the best parameters from the paper), are as follows:

|        | Train | Test | Valid |
|---|---|---|---|
| Before | 65M   | 7.1M | 403K  |
| After  | 75G   | 9.5G | 559M  |

We point this out to emphasize that if you intend to use a large dataset (such as the MOSES dataset), you need to have considerable disk space available. The sizes of these files can be reduced by specifying a larger *group_size* (default: 1000), but increasing the group size will also increase the time required for preprocessing while having a small effect on decreasing the training time.

Training and Generation jobs using the above dataset generally require <10 GB GPU memory. A model can be fully trained on MOSES after around 5 days of training on a single GPU (using a batch size of 1000).

When selecting a dataset to study, thus keep in mind that more structures in your dataset means 1) more disk space will be required to save processed dataset splits and 2) more computational time will be required for training. The number of structures should not have a significant effect on the RAM requirements of a job, as this can be controlled by the batch and block sizes used. However, the number of atom types present in the dataset will have an effect on the memory and disk space requirements of a job, as this is directly correlated to the sizes of the node and edge features tensors, as well as the sizes of the APDs. As such, you might not want to use the entire periodic table in your generative models.

Finally, as all molecules are padded up to the size of the largest graph in the dataset during Preprocessing jobs, if you have a dataset where most molecules have fewer nodes than *N*, and you have only a few structures where the number of nodes is >>*N*, a good strategy to reduce the computational requirements for this dataset would be to simply remove all molecules with >*N* nodes. The same thing could be said for the atom types and formal charges. We recommend to only keep any "outliers" in a dataset if they are deemed essential.

To summarize,

Increases disk space requirement:
* more molecules in dataset
* more atom types present in dataset
* more formal charges present
* larger molecules in dataset (really, larger *max_n_nodes*)
* smaller group size

Increases RAM:
* using a larger batch size
* using a larger block size

Increases run time:
* more molecules in dataset
* using a smaller batch size
* larger group size (Preprocessing jobs only)

Hopefully these guidelines help you in selecting an appropriate dataset to study using GraphINVENT.

### Preparing a new dataset
Once you have selected a dataset to study, you must prepare it so that it agrees with the format expected by the program. GraphINVENT expects, for each dataset, three splits in SMILES format. Each split should be named as follows:

* *train.smi*
* *test.smi*
* *valid.smi*

These should contain the training set, test set, and validation set, respectively. It is not important for the SMILES to be canonical, and it also does not matter if the file has a header or not. How many structures you put in each split is also up to you (generally the training set is larger than the testing and validation set).

You should then create a new directory in [../data/](../data/) where the name of this directory corresponds to a unique name for your dataset:

```
mkdir path/to/GraphINVENT/data/your_dataset_name/
mv train.smi valid.smi test.smi path/to/GraphINVENT/data/your_dataset_name/.
```

You will want to replace *your_dataset_name* above with the actual name for your dataset (e.g. *ChEMBL_subset*, *DRD2_actives*, etc).


### Preprocessing the new dataset
Once you have prepared your dataset in the aforementioned format, you can move on to preprocessing it using GraphINVENT. To preprocess it, you will need to know the following information:

* *max_n_nodes*
* *atom_types*
* *formal_charge*

We have provided a few scripts to help you calculate these properties in [../tools/](../tools/).

Once you know these values, you can move on to preparing a submission script. A sample submission script [../submit.py](../submit.py) has been provided. Begin by modifying the submission script to specify where the dataset can be found and what type of job you want to run. For preprocessing a new dataset, you can use the settings below, substituting in your own values where necessary:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "your_dataset_name"  # this is the dataset name, which corresponds to the directory containing the data, located in GraphINVENT/data/
job_type = "preprocess"        # this tells the code that this is a preprocessing job
jobdir_start_idx = 0           # this is an index used for labeling the first job directory where output will be written
n_jobs = 1                     # if you want to run multiple jobs (not recommended for preprocessing), set this to >1
restart = False                # this tells the code that this is not a restart job
force_overwrite = False        # if `True`, this will overwrite job directories which already exist with this name (recommend `True` only when debugging)
jobname = "preprocess"         # this is the name of the job, to be used in labeling directories where output will be written
```

Then, specify whether you want the job to run using [SLURM](https://slurm.schedmd.com/overview.html). In the example below, we specify that we want the job to run as a regular process (i.e. no SLURM). In such cases, any specified run time and memory requirements will be ignored by the script. Note: if you want to use a different scheduler, this can be easily changed in the submission script (search for "sbatch" and change it to your scheduler's submission command).

```
submit.py >
# if running using SLURM, specify the parameters below
use_slurm = False        # this tells the code to NOT use SLURM
run_time = "1-00:00:00"  # d-hh:mm:ss (will be ignored here)
mem_GB = 20              # memory in GB (will be ignored here)
```

Then, specify the path to the Python binary in the GraphINVENT virtual environment. You probably won't need to change *graphinvent_path* or *data_path*, unless you want to run the code from a different directory.

```
submit.py >
# set paths here
python_path = f"{home}/miniconda3/envs/graphinvent/bin/python"  # this is the path to the Python binary to use (change to your own)
graphinvent_path = f"./graphinvent/"                            # this is the directory containing the source code
data_path = f"./data/"                                          # this is the directory where all datasets are found
```

Finally, details regarding the specific dataset you want to use need to be entered:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],  # <-- change to your dataset's atom types
    "formal_charge": [-1, 0, +1],              # <-- change to your dataset's formal charges
    "chirality": ["None", "R", "S"],           # <-- ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 13,                         # <-- change to your dataset's value
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
}
```

At this point, you are done editing the *submit.py* file and are ready to submit a preprocesing job. You can submit the job from the terminal using the following command:

```
(graphinvent)$ python submit.py
```

During preprocessing jobs, the following will be written to the specified *dataset_dir*:
* 3 HDF files (*train.h5*, *valid.h5*, and *test.h5*)
* *preprocessing_params.csv*, containing parameters used in preprocessing the dataset (for later reference)
* *train.csv*, containing training set properties (e.g. histograms of number of nodes per molecule, number of edges per node, etc)

A preprocessing job can take a few seconds to a few hours to finish, depending on the size of your dataset. Once the preprocessing job is done and you have the above files, you are ready to run a training job using your processed dataset.

### Training models using the new dataset
You can modify the same *submit.py* script to instead run a training job using your dataset. Begin by changing the *job_type* and *jobname*; all other settings can be kept the same:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "your_dataset_name"
job_type = "train"             # this tells the code that this is a training job
jobdir_start_idx = 0
n_jobs = 1
restart = False
force_overwrite = False
jobname = "train"              # this is the name of the job, to be used in labeling directories where output will be written
```

If you would like to change the SLURM settings, you should do that next, but for this example we will keep them the same. You will then need to specify all parameters that you want to use for training:


```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],  # change to your dataset's atom types
    "formal_charge": [-1, 0, +1],              # change to your dataset's formal charges
    "chirality": ["None", "R", "S"],           # ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 13,                         # change to your dataset's value
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",                           # <-- which model to use (GGNN is the default, but showing it here to be explicit)
    "sample_every": 2,                         # <-- how often you want to sample/evaluate your model during training (for larger datasets, we recommend sampling more often)
    "init_lr": 1e-4,                           # <-- tune the initial learning rate if needed
    "epochs": 100,                             # <-- how many epochs you want to train for (you can experiment with this)
    "batch_size": 1000,                        # <-- tune the batch size if needed
    "block_size": 100000,                      # <-- tune the block size if needed
}
```

If any parameters are not specified in *submit.py* before running, the model will use the default values in [../graphinvent/parameters/defaults.py](../graphinvent/parameters/defaults.py), but it is not always the case that the "default" values will work well for your dataset. For instance, the parameters related to the learning rate decay are strongly dependent on the dataset used, and you might have to tune them to get optimal performance using your dataset. Depending on your system, you might also need to tune the mini-batch and/or block size so as to reduce/increase the memory requirement for training jobs.

You can then run a GraphINVENT training job from the terminal using the following command:

```
(graphinvent)$ python submit.py
```

As the models are training, you should see the progress bar updating on the terminal every epoch. The training status will be saved every epoch to the job directory, *output_{your_dataset_name}/{jobname}/job_{jobdir_start_idx}/*, which should be *output_{your_dataset_name}/train/job_0/* if you followed the settings above. Additionally, the evaluation scores will be saved every evaluation epoch to the job directory. Among the files written to this directory will be:

* *generation.log*, containing various evaluation metrics for the generated set, calculated during evaluation epochs
* *convergence.log*, containing the loss and learning rate for every epoch
* *validation.log*, containing model scores (e.g. NLLs, UC-JSD), calculated during evaluation epochs
* *model_restart_{epoch}.pth*, which are the model states for use in restarting jobs, or running generation/validation jobs with a trained model
* *generation/*, a directory containing structures generated during evaluation epochs (\*.smi), as well as information on each structure's NLL (\*.nll) and validity (\*.valid)

It is good to check the *generation.log* to verify that the generated set features indeed converge to those of the training set (first entry). If they do not, then you will have to tune the hyperparameters to get better performance. Furthermore, it is good to check the *convergence.log* to make sure the loss is smoothly decreasing during training.

#### Restarting a training job
If for any reason you want to restart a training job from a previous epoch (e.g. you cancelled a training job before it reached convergence), then you can do this by setting *restart = True* in *submit.py* and rerunning. While it is possible to change certain parameters in *submit.py* before rerunning (e.g. *init_lr* or *epochs*), parameters related to the model should not be changed, as the program will load an existing model from the last saved *model_restart_{epoch}.pth* file (hence there will be a mismatch between the previous parameters and those you changed). Similarly, any settings related to the file location or job name should not be changed, as the program uses those settings to search in the right directory for the previously saved model. Finally, parameters related to the dataset (e.g. *atom_types*) should not be changed, not only for a restart job but throughout the entire workflow of a dataset. If you want to use different features in the node and edge feature representations, you will have to create a copy of the dataset in [../data/](../data/), give it a unique name, and preprocess it using the desired settings.

### Generating structures using the newly trained models
Once you have trained a model, you can use a saved state (e.g. *model_restart_100.pth*) to generate molecules. To do this, *submit.py* needs to be updated to specify a generation job. The first setting that needs to be changed is the *job_type*; all other settings here should be kept fixed so that the program can find the correct job directory:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "your_dataset_name"
job_type = "generate"          # this tells the code that this is a generation job
jobdir_start_idx = 0
n_jobs = 1
restart = False
force_overwrite = False
jobname = "train"              # don't change the jobname, or the program won't find the saved model
```

You will then need to update the *generation_epoch* and *n_samples* parameter in *submit.py*:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],  # change to your dataset's atom types
    "formal_charge": [-1, 0, +1],              # change to your dataset's formal charges
    "chirality": ["None", "R", "S"],           # ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 13,                         # change to your dataset's value
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 2,                         # how often you want to sample/evaluate your model during training (for larger datasets, we recommend sampling more often)
    "init_lr": 1e-4,                           # tune the initial learning rate if needed
    "epochs": 100,                             # how many epochs you want to train for (you can experiment with this)
    "batch_size": 1000,                        # <-- tune the batch size if needed
    "block_size": 100000,                      # tune the block size if needed
    "generation_epoch": 100,                   # <-- specify which saved model (i.e. at which epoch) to use for training)
    "n_samples": 30000,                        # <-- specify how many structures you want to generate
}
```

The *generation_epoch* should correspond to the saved model state that you want to use for generation, and *n_samples* tells the program how many structures you want to generate. In the example above, the parameters specify that the model saved at Epoch 100 should be used to generate 30,000 structures. All other parameters should be kept the same (if they are related to training, such as *epochs* or *init_lr*, they will be ignored during generation jobs).

Structures will be generated in batches of size *batch_size*. If you encounter memory problems during generation jobs, reducing the batch size should once again solve them. Generated structures, along with their corresponding metadata, will be written to the *generation/* directory within the existing job directory. These files are:

* *epochGEN{generation_epoch}_{batch}.smi*, containing molecules generated at the epoch specified
* *epochGEN{generation_epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{generation_epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.log* file will be updated with the various evaluation metrics for the generated structures.

If you've followed the tutorial up to here, it means you can successfully create new molecules using a GNN-based model trained on a custom dataset.

#### (Optional) Postprocessing

To make things more convenient for any subsequent analyses, you can concatenate all structures generated in different batches into one file using:

```
for i in epochGEN{generation_epoch}_*.smi; do cat $i >> epochGEN{generation_epoch}.smi; done
```

Above, *{generation_epoch}* should be replaced with a number corresponding to a valid epoch. You can do similar things for the NLL and validity files, as the rows in those files correspond to the rows in the SMILES files.

Note that "Xe" and empty graphs may appear in the generated structures, even if the models are well-trained, as there is always a small probability of sampling invalid actions. If you do not want to include invalid entries in your analysis, these can be filtered out by typing:

```
sed -i "/Xe/d" path/to/file.smi          # remove "Xe" placeholders from file
sed -i "/^ [0-9]\+$/d" path/to/file.smi  # remove empty graphs from file
```

See [3_visualizing_molecules](./3_visualizing_molecules.md) for examples on how to draw grids of molecules.

### A note about hyperparameters
If you've reached this part of the tutorial, you now have a good idea of how to train GraphINVENT models on custom datasets. Nonetheless, as hinted above, some hyperparameters are highly dependent on the dataset used, and you may have to do some hyperparameter tuning to obtain the best performance using your specific dataset. In particular, parameters related to the learning rate decay are sensitive to the dataset, so a bit of experimentation here is recommended when using a new dataset as these parameters can make a difference between an "okay" model and a well-trained model. These parameters are:

* *init_lr*
* *min_rel_lr*
* *lrdf*
* *lrdi*

If any parameters are not specified in the submission script, the program will use the default values from [../graphinvent/parameters/defaults.py](../graphinvent/parameters/defaults.py). Have a look there if you want to learn more about any additional hyperparameters that may not have been discussed in this tutorial. Note that not all parameters defined in *../graphinvent/parameters/defaults.py* are model-related hyperparameters; many are simply practical parameters and settings, such as the path to the datasets being studied.

### Summary
Hopefully you are now able to train models on custom datasets using GraphINVENT. If anything is unclear in this tutorial, or if you have any questions that have not been addressed by this guide, feel free to contact the authors for assistance. Note that a lot of useful information centered about hyperparameter tuning is available in our [technical note](https://chemrxiv.org/articles/preprint/Practical_Notes_on_Building_Molecular_Graph_Generative_Models/12888383/1).

We look forward to seeing the molecules you've generated using GraphINVENT.
