# load general packages and functions
import csv
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

# functions for loading SMILES and model type



def molecules(path):
    """ Reads a SMILES file (full path/filename specified by `path`) and
    returns a `list` of `rdkit.Mol` objects.
    """
    # check first line of SMILES file to see if contains header
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=has_header)

    return molecule_set


def which_model(input_csv_path):
    """ Gets the type of model to use by reading it from CSV (in "input.csv").
    """
    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=";")

        for key, value in params_reader:
            if key == "model":
                return value  # string describing model e.g. "GGNN"

