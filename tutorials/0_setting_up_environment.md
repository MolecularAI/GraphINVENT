## Setting up the environment
Before doing anything with GraphINVENT, you will need to configure the GraphINVENT virtual environment, as the code is dependent on very specific versions of packages. You can use [conda](https://docs.conda.io/en/latest/) for this.

The [../environments/graphinvent.yml](../environments/graphinvent.yml) file lists all the packages required for GraphINVENT to run. From within the [GraphINVENT/](../) directory, a virtual environment can be easily created using the YAML file and conda by typing into the terminal:

```
conda env create -f environments/graphinvent.yml
```

Then, to activate the environment:

```
conda activate graphinvent
```

To install additional packages to the virtual environment, should the need arise, use:

```
conda install -n graphinvent {package_name}
```

To save an updated environment as a YAML file using conda, use:

```
conda env export > path/to/environment.yml
```

And that's it! To learn how to start training models, go to [1_introduction](1_introduction.md).
