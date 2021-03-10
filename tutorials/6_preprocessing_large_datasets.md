## Preprocessing large datasets

For preprocessing very large datasets (e.g. MOSES, with over 1M structures in the training set), it is recommended to split up the data and preprocess them on separate CPUs.

Until I get around to fixing a way to do this in the code, one can do it the hacky way. In the hacky way, we simply split up the large dataset into many smaller datasets, preprocess them as separate CPU jobs, and then combine them with a hacky script at the end.

So first, split up the desired SMILES file by running

```
split -l 100000 train.smi
```

The above line is of course assuming that you want to split the training data.

Then, place each of the splits in a separate directory in [../data/](../data/), such as *my_dataset_1/train.smi*, and make sure to rename them into "train.smi" from whatever the split output is (e.g. "xaa", "xab", etc).

Then, comment out the lines for preprocessing the validation and test sets in [../graphinvent/Workflow.py](../graphinvent/Workflow.py):

```
# self.preprocess_valid_data()                                        
# self.preprocess_test_data()
```

Finally, set your desired parameters in *submit.py* and run a preprocessing job for each split (within the GraphINVENT conda environment):

```
(graphinvent)$ python submit.py
```

Once all the HDF files are preprocessed, these can be combined using [../tools/combine_HDFs.py](../tools/combine_HDFs.py).

Don't forget to uncomment out the above lines in the future.

