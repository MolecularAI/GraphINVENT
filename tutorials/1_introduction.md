## Introduction to GraphINVENT
As shown in our recent [publication]((https://chemrxiv.org/articles/preprint/Graph_Networks_for_Molecular_Design/12843137/1)), GraphINVENT can be used to learn the structure and connectivity of sets of molecular graphs, thus making it a promising tool for the generation of molecules resembling an input dataset. As models in GraphINVENT are probabilistic, they can be used to discover new molecules that are not present in the training set.

There are six GNN-based models implemented in GraphINVENT: the MNN, GGNN, AttGGNN, S2V, AttS2V, and EMN models. The GGNN has shown the best performance when weighed against the computational time required for training, and is as such used as the default model.

To begin using GraphINVENT, we have prepared the following tutorial to guide a new user through the molecular generation workflow using a small example dataset. The example dataset is a 1K random subset of GBD-13. It has already been preprocessed, so you can use it directly for Training and Generation, as we will show in this tutorial. If this is too simple and you would like to learn how to train GraphINVENT models using a new molecular dataset, see [2_using_a_new_dataset](./2_using_a_new_dataset.md).

### Training using the example dataset
#### Preparing a training job
The example dataset is located in [../GraphINVENT/data/gdb13_1K/](../GraphINVENT/data/gdb13_1K/) and contains the following:
* 1K molecules in each the training, validation, and test set
* atom types : {C, N, O, S, Cl}
* formal charges : {-1, 0, +1}
* max num nodes : 13 (it is a subset of GDB-13).

The last three points of information must be included in the submission script, as well as any additional parameters and hyperparameters to use for the training job.

A sample submission script [submit.py](../submit.py) has been provided. Begin by modifying the submission script to specify where the dataset can be found and what type of job you want to run. For training on the example set, the settings below are recommended:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "gdb13_1K"     # this is the dataset name, which corresponds to the directory containing the data, located in GraphINVENT/data/
job_type = "train"       # this tells the code that this is a training job
jobdir_start_idx = 0     # this is an index used for labeling the first job directory where output will be written
n_jobs = 1               # if you want to run multiple jobs (e.g. for collecting statistics), set this to >1
restart = False          # this tells the code that this is not a restart job
force_overwrite = False  # if `True`, this will overwrite job directories which already exist with this name (recommend `True` only when debugging)
jobname = "example"      # this is the name of the job, to be used in labeling directories where output will be written
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

Finally, details regarding the specific dataset and parameters you want to use need to be entered. If they are not specified in *submit.py* before running, the model will use the default values in [./graphinvent/parameters/defaults.py](./graphinvent/parameters/defaults.py), but it is not always the case that the "default" values will work well for your dataset. The models are sensitive to the hyperparameters used for each dataset, especially the learning rate and learning rate decay. For the example dataset, the following parameters are recommended:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, +1],
    "max_n_nodes": 13,
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 10,
    "init_lr": 1e-4,     # (!)
    "epochs": 400,
    "batch_size": 1000,
    "block_size": 100000,
}
```

Above, (!) indicates that a parameter is strongly dependent on the dataset used. Note that, depending on your system, you might need to tune the mini-batch and/or block size so as to reduce/increase the memory requirement for training jobs. There is an inverse relationship between the batch size and the time required to train a model. As such, only reduce the batch size if necessary, as decreasing the batch size will lead to noticeably slower training.

At this point, you are done editing the *submit.py* file and are ready to submit a training job.

#### Running a training job
Using the prepared *submit.py*, you can run a GraphINVENT training job from the terminal using the following command:

```
(graphinvent)$ python submit.py
```

Note that for the code to run, you need to have configured and activated the GraphINVENT environment (see [0_setting_up_environment](0_setting_up_environment.md) for help with this).

As the models are training, you should see the progress bar updating on the terminal every epoch. The training status will be saved every epoch to the job directory, *output_{dataset}/{jobname}/job_{jobdir_start_idx}/*, which should be *output_gdb13_1K/example/job_0/* if you followed the settings above. Additionally, the evaluation scores will be saved every evaluation epoch to the job directory. Among the files written to this directory will be:

* *generation.log*, containing various evaluation metrics for the generated set, calculated during evaluation epochs
* *convergence.log*, containing the loss and learning rate for every epoch
* *validation.log*, containing model scores (e.g. NLLs, UC-JSD), calculated during evaluation epochs
* *model_restart_{epoch}.pth*, which are the model states for use in restarting jobs, or running generation/validation jobs with a trained model
* *generation/*, a directory containing structures generated during evaluation epochs (\*.smi), as well as information on each structure's NLL (\*.nll) and validity (\*.valid)

It is good to check the *generation.log* to verify that the generated set features indeed converge to those of the training set (first entry). If they do not then something is wrong (most likely bad hyperparameters). Furthermore, it is good to check the *convergence.log* to make sure the loss is smoothly decreasing during training.

#### Restarting a training job
If for any reason you want to restart a training job from a previous epoch (e.g. you cancelled a training job before it reached convergence), then you can do this by setting *restart = True* in *submit.py* and rerunning. While it is possible to change certain parameters in *submit.py* before rerunning (e.g. *init_lr* or *epochs*), parameters related to the model should not be changed, as the program will load an existing model from the last saved *model_restart_{epoch}.pth* file (hence there will be a mismatch between the previous parameters and those you changed). Similarly, any settings related to the file location or job name should not be changed, as the program uses those settings to search in the right directory for the previously saved model. Finally, parameters related to the dataset (e.g. *atom_types*) should not be changed, not only for a restart job but throughout the entire workflow of a dataset. If you want to use different features in the node and edge feature representations, you will have to create a copy of the dataset in [../data/](../data/), give it a unique name, and preprocess it using the desired settings.

### Generation using a trained model
#### Running a generation job
Once you have trained a model, you can use a saved state (e.g. *model_restart_400.pth*) to generate molecules. To do this, *submit.py* needs to be updated to specify a generation job. The first setting that needs to be changed is the *job_type*; all other settings here should be kept fixed so that the program can find the correct job directory:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "gdb13_1K"
job_type = "generate"    # this tells the code that this is a generation job
jobdir_start_idx = 0
n_jobs = 1
restart = False
force_overwrite = False
jobname = "example"
```

You will then need to update the *generation_epoch* and *n_samples* parameters in *submit.py*:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, +1],
    "max_n_nodes": 13,
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 10,
    "init_lr": 1e-4,          # (!)
    "epochs": 400,
    "batch_size": 1000,
    "block_size": 100000,
    "generation_epoch": 400,  # <-- which model to use (i.e. which epoch)
    "n_samples": 30000,       # <-- how many structures to generate
}
```

The *generation_epoch* should correspond to the saved model state that you want to use for generation. In the example above, the parameters specify that the model saved at Epoch 400 should be used to generate 30,000 molecules. All other parameters should be kept the same (if they are related to training, such as *epochs* or *init_lr*, they will be ignored during generation jobs).

Structures will be generated in batches of size *batch_size*. If you encounter memory problems during generation jobs, reducing the batch size should once again solve them. Generated structures, along with their corresponding metadata, will be written to the *generation/* directory within the existing job directory. These files are:

* *epochGEN{generation_epoch}_{batch}.smi*, containing molecules generated at the epoch specified
* *epochGEN{generation_epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{generation_epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.log* file will be updated with the various evaluation metrics for the generated structures.

If you've followed the tutorial up to here, it means you can successfully create new molecules using a trained GNN-based model.

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

### Summary
Now you know how to train models and generate structures using the example dataset. However, the example dataset structures are not drug-like, and are therefore not the most interesting to study for drug discovery applications. To learn how to train GraphINVENT models on custom datasets, see [2_using_a_new_dataset](./2_using_a_new_dataset.md).
