### Setting up the environment
Before doing anything with GraphINVENT, you will need to configure the GraphINVENT environment, as the code is dependent on very specific versions of packages. We will use conda for this.

We have provided a *GraphINVENT-env.yml* file in *environments/*, which lists all the packages along with their exact versions which are required for GraphINVENT to run. From within the *GraphINVENT/* directory, a virtual environment can be easily created from the YAML file by typing into the terminal:

```
conda env create -f environments/GraphINVENT-env.yml
```

Then, to activate the environment:

```
conda activate GraphINVENT-env
```

To install additional packages to the virtual environment, should the need arise, use:

```
conda install -n GraphINVENT-env {package_name}
```

To save an updated environment as a YAML file using conda, use:
```
conda env export > path/to/environment.yml
```