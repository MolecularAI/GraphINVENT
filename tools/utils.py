"""
Miscellaneous functions.
"""
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier


def load_molecules(path : str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:
    """
    Reads a SMILES file (full path/filename specified by `path`) and returns the
    `rdkit.Mol` object "supplier".
    """
    # check first line of SMILES file to see if contains header
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=has_header)

    return molecule_set
