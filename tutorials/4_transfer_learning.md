## Transfer learning using GraphINVENT
In this tutorial, you will be guided through the process of generating molecules with targeted properties using transfer learning (TL).

This tutorial assumes that you have looked through tutorials [1_introduction](./1_introduction.md) and [2_using_a_new_dataset](./2_using_a_new_dataset.md).

### Selecting two (or more) datasets
In order to do transfer learning, you must first select two datasets which you would like to work with. The first and (probably) larger dataset should be one that you can use to train your model generally, whereas the second should be one containing (a few) examples of molecules exhibiting the properties you desire in your generated molecules (e.g. known actives).

When choosing your datasets, first, remember that GraphINVENT models are computationally demanding; I recommend you go back and review the *Selecting a new dataset* guidelines provided in [2_using_a_new_dataset](./2_using_a_new_dataset.md).

Second, ideally there is some amount of overlap between the structures in your general training set (set 1) and your targeted training set (set 2). If the two sets are totally different, it will be difficult for your model to learn how to apply what it learns from set 1 to set 2. However, they also should not come from the exact same distributions (otherwise, what's the point of doing TL...). 


### Preparing a new dataset
Once you have chosen your two datasets, you must prepare them so that they agree with the format expected by the program. GraphINVENT expects, for each dataset, three splits in SMILES format. Each split should be named as follows:

* *train.smi*
* *test.smi*
* *valid.smi*

These should contain the training set, test set, and validation set, respectively. It is not important for the SMILES to be canonical, and it also does not matter if the file has a header or not. How many structures you put in each split is also up to you.

You should then create two new directories in [../data/](../data/), one for each dataset, where the name of each directory corresponds to a unique name for the dataset it contains:

```
mkdir path/to/GraphINVENT/data/set_1/
./split_dataset set_1.smi  # example script that writes a train.smi, valid.smi, and test.smi from set_1.smi
mv train.smi valid.smi test.smi path/to/GraphINVENT/data/set_1/.

mkdir path/to/GraphINVENT/data/set_2/
./split_dataset set_2.smi  # example script that writes a train.smi, valid.smi, and test.smi from set_2.smi
mv train.smi valid.smi test.smi path/to/GraphINVENT/data/set_2/.

```

You will want to replace *set_1* and *set_2* above with the actual names for your datasets (e.g. *ChEMBL_subset*, *DRD2_actives*, etc).


### Preprocessing the new dataset
Once you have prepared your datasets in the aforementioned format, you can move on to preprocessing them using GraphINVENT. To preprocess them, you will need to know the following information:

* *max_n_nodes*
* *atom_types*
* *formal_charge*

Be careful to calculate this for BOTH sets, and not just one e.g. if the *max_n_nodes* in set 1 is 38, and the *max_n_nodes* in set 2 is 15, then the *max_n_nodes* for BOTH sets will be 38. Similarly, if the *atom_types* in set 1 are ["C", "N", "O"] and the *atom_types* in set 2 are ["C", "O", "S"], then the *atom_types* for BOTH sets will be ["C", "N", "O", "S"]. Here, the specific order of elements in *atom_types* does not matter, so long as the order is the same for BOTH sets.

We have provided a few scripts to help you calculate these properties in [../tools/](../tools/).

Once you know these values, you can move on to preparing a submission script for preprocessing the first dataset. A sample submission script [../submit.py](../submit.py) has been provided. Begin by modifying the submission script to specify where the dataset can be found and what type of job you want to run. For preprocessing a new dataset, you can use the settings below, substituting in your own values where necessary:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "set 1"              # this is the dataset name, which corresponds to the directory containing the data, located in GraphINVENT/data/
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
python_path = f"../miniconda3/envs/graphinvent/bin/python"  # this is the path to the Python binary to use (change to your own)
graphinvent_path = f"./graphinvent/"                            # this is the directory containing the source code
data_path = f"./data/"                                          # this is the directory where all datasets are found
```

Finally, details regarding the specific dataset you want to use need to be entered. Here, you must remember to use the properties for BOTH datasets:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S"],        # <-- change to your datasets' atom types
    "formal_charge": [-1, 0, +1],              # <-- change to your datasets' formal charges
    "chirality": ["None", "R", "S"],           # <-- ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 38,                         # <-- change to your datasets' value
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

A preprocessing job can take a few seconds to a few hours to finish, depending on the size of your dataset.

Once you have preprocessed the first dataset, you must go back and preprocess the second dataset. To do this, you can use the same *submit.py* file; simply go back and change the dataset name:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "set 2"              # this is the dataset name, which corresponds to the directory containing the data, located in GraphINVENT/data/ <-- this line changed
job_type = "preprocess"        # this tells the code that this is a preprocessing job
jobdir_start_idx = 0           # this is an index used for labeling the first job directory where output will be written
n_jobs = 1                     # if you want to run multiple jobs (not recommended for preprocessing), set this to >1
restart = False                # this tells the code that this is not a restart job
force_overwrite = False        # if `True`, this will overwrite job directories which already exist with this name (recommend `True` only when debugging)
jobname = "preprocess"         # this is the name of the job, to be used in labeling directories where output will be written
```

...and re-run:

```
(graphinvent)$ python submit.py
```

Once you have preprocessed both datasets, you are ready to run a general training job using the first dataset.

### Training models generally
You can modify the same *submit.py* script to instead run a training job using the general dataset (set 1). Begin by changing the *job_type* and *jobname*; all other settings can be kept the same:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "set_1"
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
    "atom_types": ["C", "N", "O", "S"],        # change to your datasets' atom types
    "formal_charge": [-1, 0, +1],              # change to your datasets' formal charges
    "chirality": ["None", "R", "S"],           # ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 38,                         # change to your datasets' value
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

If any parameters are not specified in *submit.py* before running, the model will use the default values in [../graphinvent/parameters/defaults.py](../graphinvent/parameters/defaults.py), but it is not always the case that the "default" values will work well for your datasets. For instance, the parameters related to the learning rate decay are strongly dependent on the dataset used, and you might have to tune them to get optimal performance using your datasets. Depending on your system, you might also need to tune the mini-batch and/or block size so as to reduce/increase the memory requirement for training jobs.

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

### Fine-tuning your predictions
By this point, you have trained a model on a general dataset, so it has ideally learned how to form chemically valid compounds. However, the next thing we would like to do is fine-tune the models on a smaller set of molecules possessing the molecular properties that we would like our generated molecules to have. To do this, we can resume training from a generally trained model.

To do this, we can once again modify *submit.py* to specify a restart job on the second dataset:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "set_1"              # <-- change this from set 1 to set 2
job_type = "train"             # this tells the code that this is a training job
jobdir_start_idx = 0
n_jobs = 1
restart = True                 # <-- specify a restart job
force_overwrite = False
jobname = "train"              # this is the name of the job, to be used in labeling directories where output will be written (don't change this! otherwise GraphINVENT won't find the saved model states)
```

At this point, you can also fine-tune the training parameters, but below we have chosen to keep them all the same (you will have to see what works and what doesn't work for your dataset):

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S"],        # change to your datasets' atom types
    "formal_charge": [-1, 0, +1],              # change to your datasets' formal charges
    "chirality": ["None", "R", "S"],           # ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 38,                         # change to your datasets' value
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

Before submitting, you must also create a new output directory (manually) for set 2 containing the saved model state and the *convergence.log* file, following the same directory structure as the output directory for set 1:

```
mkdir output_set_2/train/job_0/
cp output_set_2/train/job_0/model_restart_100.pth output_set_2/train/job_0/.
cp output_set_2/train/job_0/convergence.log output_set_2/train/job_0/.
```

This is necessary in order for GraphINVENT to successfully find the previous saved model state, containing the "generally" trained model.

Once you have done this, you can run the new training job from the terminal using the following command:

```
(graphinvent)$ python submit.py
```

The job will restart from the last saved state, so, for example, if your first training job on set 1 reached Epoch 100, then training on set 2 will resume at the model state saved then.

### Generating structures using the fine-tuned model
Once you have fine-tuned your model, you can use a saved state (e.g. *model_restart_200.pth*) to generate targeted molecules. To do this, *submit.py* needs to be updated to specify a generation job. The first setting that needs to be changed is the *job_type*; all other settings here should be kept fixed so that the program can find the correct job directory:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "set_2"
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
    "atom_types": ["C", "N", "O", "S"],        # change to your dataset's atom types
    "formal_charge": [-1, 0, +1],              # change to your dataset's formal charges
    "chirality": ["None", "R", "S"],           # ignored, unless you also specify `use_chirality`=True
    "max_n_nodes": 38,                         # change to your dataset's value
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 2,                         # how often you want to sample/evaluate your model during training (for larger datasets, we recommend sampling more often)
    "init_lr": 1e-4,                           # tune the initial learning rate if needed
    "epochs": 200,                             # how many epochs you want to train for (you can experiment with this)
    "batch_size": 1000,                        # <-- tune the batch size if needed
    "block_size": 100000,                      # tune the block size if needed
    "generation_epoch": 100,                   # <-- specify which saved model (i.e. at which epoch) to use for training)
    "n_samples": 30000,                        # <-- specify how many structures you want to generate
}
```

The *generation_epoch* should correspond to the saved model state that you want to use for generation, and *n_samples* tells the program how many structures you want to generate. In the example above, the parameters specify that the model saved at Epoch 200 should be used to generate 30,000 structures. All other parameters should be kept the same (if they are related to training, such as *epochs* or *init_lr*, they will be ignored during generation jobs).

Structures will be generated in batches of size *batch_size*. If you encounter memory problems during generation jobs, reducing the batch size should once again solve them. Generated structures, along with their corresponding metadata, will be written to the *generation/* directory within the existing job directory. These files are:

* *epochGEN{generation_epoch}_{batch}.smi*, containing molecules generated at the epoch specified
* *epochGEN{generation_epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{generation_epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.log* file will be updated with the various evaluation metrics for the generated structures.

If you've followed the tutorial up to here, it means you can successfully create new, targeted molecules using transfer learning.

Please see the other tutorials (e.g. [1_introduction](./1_introduction.md) and [2_using_a_new_dataset](./2_using_a_new_dataset.md)) for details on how one can post-process the structures for easy visualization, as well as how one can tune the hyperparameters to improve model performance using the different datasets.

### Summary
Hopefully you are now able to train models to generate targeted molecules using transfer learning in GraphINVENT. If anything is unclear in this tutorial, or if you have any questions that have not been addressed by this guide, feel free to contact [me](https://github.com/rociomer).
