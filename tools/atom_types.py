"""
Gets the atom types present in a set of molecules.

To use script, run:
python atom_types.py --smi path/to/file.smi
"""
import argparse
import rdkit
from utils import load_molecules


# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--smi",
                    type=str,
                    default="data/gdb13_1K/train.smi",
                    help="SMILES file containing molecules to analyse.")
args = parser.parse_args()


def get_atom_types(smi_file : str) -> list:
    """
    Determines the atom types present in an input SMILES file.

    Args:
    ----
        smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)

    # create a list of all the atom types
    atom_types = list()
    for mol in molecules:
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())

    # remove duplicate atom types then sort by atomic number
    set_of_atom_types = set(atom_types)
    atom_types_sorted = list(set_of_atom_types)
    atom_types_sorted.sort()

    # return the symbols, for convenience
    return [rdkit.Chem.Atom(atom).GetSymbol() for atom in atom_types_sorted]


if __name__ == "__main__":
    atom_types = get_atom_types(smi_file=args.smi)
    print("* Atom types present in input file:", atom_types, flush=True)
    print("Done.", flush=True)
