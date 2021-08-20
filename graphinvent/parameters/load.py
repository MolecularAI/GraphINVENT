"""
Functions for loading molecules from SMILES, as well as loading the model type.
"""
# load general packages and functions
import csv
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def molecules(path : str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:
    """
    Reads a SMILES file (full path/filename specified by `path`) and returns
    `rdkit.Mol` objects.
    """
    # check first line of SMILES file to see if contains header
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path,
                                     sanitize=True,
                                     nameColumn=-1,
                                     titleLine=has_header)
    return molecule_set

def which_model(input_csv_path : str) -> str:
    """
    Gets the type of model to use by reading it from CSV (in "input.csv").

    Args:
    ----
        input_csv_path (str) : The full path/filename to "input.csv" file
          containing parameters to overwrite from defaults.

    Returns:
    -------
        value (str) : Name of model to use.
    """
    with open(input_csv_path, "r") as csv_file:

        params_reader = csv.reader(csv_file, delimiter=";")

        for key, value in params_reader:
            if key == "model":
                return value  # string describing model e.g. "GGNN"

    raise ValueError("Model type not specified.")
