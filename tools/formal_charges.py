"""
Gets the formal charges present in a set of molecules.

To use script, run:
python formal_charges.py --smi path/to/file.smi
"""
import argparse
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


def get_formal_charges(smi_file : str) -> list:
    """
    Determines the formal charges present in an input SMILES file.

    Args:
    ----
        smi_file (str) : Full path/filename to SMILES file.
    """
    molecules = load_molecules(path=smi_file)

    # create a list of all the formal charges
    formal_charges = list()
    for mol in molecules:
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

    # remove duplicate formal charges then sort
    set_of_formal_charges = set(formal_charges)
    formal_charges_sorted = list(set_of_formal_charges)
    formal_charges_sorted.sort()

    return formal_charges_sorted


if __name__ == "__main__":
    formal_charges = get_formal_charges(smi_file=args.smi)
    print("* Formal charges present in input file:", formal_charges, flush=True)
    print("Done.", flush=True)
