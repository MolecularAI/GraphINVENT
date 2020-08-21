# load general packages and functions
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier



def load_molecules(path):
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

def get_max_n_atoms(smi_file):
    """ Determines the maximum number of atoms per molecule in an input SMILES file.

    Args:
      smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)

    max_n_atoms = 0
    for mol in molecules:
        n_atoms = mol.GetNumAtoms()

        if n_atoms > max_n_atoms:
            max_n_atoms = n_atoms

    return max_n_atoms


if __name__ == "__main__":
    max_n_atoms = get_max_n_atoms(smi_file="/path/to/SMILES.smi")  # e.g. smi_file="data/gdb13_1K/train.smi"
    print("* Max number of atoms in input dataset:", max_n_atoms, flush=True)
    print("Done.", flush=True)
