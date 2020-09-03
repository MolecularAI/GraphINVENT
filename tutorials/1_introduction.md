### Introduction to GraphINVENT
GraphINVENT can be used to generate novel molecules based on an input dataset. There are six different GNN-based models implemented in GraphINVENT: the MNN, GGNN, AttGGNN, S2V, AttS2V, and EMN models. The GGNN has shown the best performance when weighed against the computational time required for training.

To begin using GraphINVENT, we have prepared the following tutorial to guide a new user through the molecular generation workflow using a small example dataset. The example dataset is a 1K random subset of GBD-13. It has already been preprocessed, so the user can use it directly for Training and Generation. To learn how to train GraphINVENT using a new molecular dataset, see *2_using_a_new_dataset.md*.

### Training using the example dataset
#### Preparing training job
The example dataset is located in *GraphINVENT/data/gdb13_1K/* and contains the following:
* 1K molecules in each the training, validation, and test set
* atom types : {C, N, O, S, Cl}
* formal charges : {-1, 0, +1}
* max num nodes : 13 (it is a subset of GDB-13).

The last three points of information must be included in the submission script, as well as any additional parameters and hyperparameters to use for the training job.

A sample submission script has been provided in the main GraphINVENT directory, called *submit.py*.

Begin by modifying the submission script to specify where the dataset can be found and what type of job you want to run. For training on the example set, we recommend the settings below:

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "gdb13_1K"     # this is the dataset name, which corresponds to the directory name containing the data, located in GraphINVENT/data/
job_type = "train"       # this tells the code that this is a training job
jobdir_start_idx = 0     # this is an index used for labeling the first job directory where output will be written
n_jobs = 1               # if you want to run multiple jobs (e.g. for collecting statistics), set this to >1
restart = False          # this tells the code that this is not a restart job
force_overwrite = False  # if `True`, this will overwrite job directories which already exist with this name (recommend `True` only when debugging)
jobname = "example"      # this is the name of the job, to be used in labeling directories where output will be written
```

Then, specify whether you want the job to be run as a batch job. In the example below, we specify that we want the job to run as a regular job, as we will simply run this on the workstation. In this case, the specified run time and memory will be ignored.

```
submit.py >
# if running as batch jobs, also specify the parameters below
use_sbatch = False       # this tells the code to NOT run a batch job
run_time = "1-00:00:00"  # hh:mm:ss (will be ignored here)
mem_GB = 20              # memory in GB (will be ignored here)
```

Then, you need to specify the paths to the correct Python binary. You probably will not need to change *graphinvent_path* or *data_path*, unless you want to run the code from a different directory other than the GraphINVENT/ directory.

```
submit.py >
# set paths here
python_path = f"../miniconda3/envs/GraphINVENT-env/bin/python"  # this is the path to the Python binary to use
graphinvent_path = f"./graphinvent/"                            # this is the path to the source code
data_path = f"./data/"                                          # this is the directory where all datasets can be found
```

Finally, details regarding the specific dataset and parameters you want to use need to be specified. If they are not specified in *submit.py* before running, the model will use the default values in *graphinvent/parameters/defaults.py*, but it is not always the case that the "default" values will work well for your dataset. The models are sensitive to the hyperparameters used, especially the learning rate and learning rate decay. For the example dataset, we recommend setting parameters to the following:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, +1],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 13,
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 10,
    "min_rel_lr": 5e-2,  # (!)
    "lrdf": 0.9999,      # (!)
    "lrdi": 100,         # (!)
    "init_lr": 1e-4,     # (!)
    "epochs": 400,
    "batch_size": 1000,
    "block_size": 100000,
}
```

Above, (!) indicates that a parameter is strongly dependent on the dataset used. Note that, depending on your system, you might need to tune the mini-batch and/or block size so to reduce the memory requirement for training jobs. Only do this if necessary, however, as decreasing the batch size will inevitably lead to slower training.

At this point you are done editing the *submit.py* file.

#### Running a training job
Using the prepared *submiy.py*, you can run a GraphINVENT training job from the terminal using the following command:

```
(GraphINVENT-env)$ python submit.py
```

Note that for the code to run, you will need to have set up the GraphINVENT environment and activated it (see *0_setting_up_environment.md* for help with this).

As the models are training, you should see the progress bar updating on the terminal every epoch. The training status will be saved every epoch to the job directory (*output_{dataset}/{jobname}/job_{jobdir_start_idx}*, which should be *output_gdb13_1K/example/job_0* if you followed the settings above). Additionally, the evaluation scores will be saved every evaluation epoch to the job directory. Among the files written to this directory will be:

* *generation.csv*, containing various evaluation metrics for the generated set, calculated during evaluation epochs
* *convergence.csv*, containing the loss and learning rate for every epoch
* *validation.csv*, containing model scores (e.g. NLLs, UC-JSD), calculated during evaluation epochs
* *model_restart_{epoch}.pth*, which are the model states for use in restarting jobs, or running generation/validation jobs with a trained model
* *generation/*, a directory containing structures generated during evaluation epochs (\*.smi), as well as information on each structure's NLL (\*.nll) and validity (\*.valid)

It is good to check the *generation.csv* to verify that the generated set features indeed converge to those of the training set (first entry). If they do not then something is wrong (most likely bad hyperparameters). Furthermore, it is good to check the *convergence.csv* to make sure the loss is smoothly decreasing during training.

#### Restarting a training job
If for any reason you want to restart a training job from a previous epoch (e.g. you cancelled a training job before it reached convergence), then you can do this by setting the *restart = True* in *submit.py* and rerunning. While it is possible to change certain parameters in *submit.py* before rerunning (e.g. *init_lr* or *epochs*), parameters related to the model should not be changed, as the program will load an existing model from the last saved *model_restart_{epoch}.pth* file. Similarly, any settings related to the file location or job name should not be changed, as the program won't search in the right directory for the previously saved model. Finally, parameters related to the dataset (e.g. *atom_types*) should not be changed, not only for a restart job but throughout the entire workflow of a dataset. If you want to use different features in the node and edge feature representations, you will have to create a copy of the dataset (with a different name) and reprocess it using the new settings.

### Generation using a trained model
#### Running a generation job
One you have a trained model, you can use a saved model (e.g. *model_restart_400.pth*) to generate molecules. To do this, *submit.py* needs to be updated to specify the parameters for a generation job. The only setting that needs to be changed is the *job_type*, all other settings should be kept fixed so that the program can find the correct job directory.

```
submit.py >
# define what you want to do for the specified job(s)
dataset = "gdb13_1K"
job_type = "generate"  # this tells the code that this is a generation job
jobdir_start_idx = 0
n_jobs = 1
restart = False
force_overwrite = False
jobname = "example"
```

You will then need to update the *generation_epoch* parameter in *submit.py*:

```
submit.py >
# define dataset-specific parameters
params = {
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, +1],
    "chirality": ["None", "R", "S"],
    "max_n_nodes": 13,
    "job_type": job_type,
    "dataset_dir": f"{data_path}{dataset}/",
    "restart": restart,
    "model": "GGNN",
    "sample_every": 10,
    "min_rel_lr": 5e-2,  # (!)
    "lrdf": 0.9999,      # (!)
    "lrdi": 100,         # (!)
    "init_lr": 1e-4,     # (!)
    "epochs": 400,
    "batch_size": 1000,
    "block_size": 100000,
    "generation_epoch": 400,
}
```

The *generation_epoch* should correspond to the saved model state that you want to use for generation. In the case above, we chose 400. All other parameters should be kept the same (if they are related to training, such as *epochs* or *init_lr*, they will be ignored during generation jobs).

Structures will be generated in batches of size *batch_size*. If you encounter memory problems during generation jobs, reducing the batch size should solve them. Generated structures, along with their corresponding metadata, will be written to the *generation/* directory within the existing job directory. These files are:

* *epochGEN{epoch}_{batch}.smi*, containing molecules generated at the epoch specified by *generation_epoch*
* *epochGEN{epoch}_{batch}.nll*, containing their respective NLLs
* *epochGEN{epoch}_{batch}.valid*, containing their respective validity (0: invalid, 1: valid)

Additionally, the *generation.csv* file will be updated with the various evaluation metrics for the generated structures. And, you're done! If you've followed the tutorial up to here, it means you can successfully create new molecules using a trained GNN-based model.

#### (Optional) Postprocessing

To make things more convenient for yourself, you can concatenate all the structures generated for each batch into one file using:

```
for i in epochGEN{epoch}_*smi; do cat $i >> epochGEN{epoch}.smi; done
```

Above, *{epoch}* should be replaced with a number corresponding to a valid epoch. You can do similar things for the NLL and validity files, as the rows in those files correspond to the rows in the SMILES files.

Note that "Xe" and empty graphs may appear in the generated structures, even if the models are well-trained, as there is always a small probability of sampling invalid actions. To view only the valid molecules, you can do something like:

```
sed -i "/Xe/d" epochGEN{epoch}.smi          # remove "Xe" placeholders from file
sed -i "/^ [0-9]\+$/d" epochGEN{epoch}.smi  # remove empty graphs from file
```

Structures can be visualized in RDKit. See *3_visualizing_molecules* for examples.

### Summary

