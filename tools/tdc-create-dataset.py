"""
Uses the Therapeutics Data Commons (TDC) to get datasets for goal-directed
molecular optimization tasks.

See:
* https://tdcommons.ai/
* https://github.com/mims-harvard/TDC

To use script, run:
python tdc-create-dataset.py TODO
"""
import os
import argparse
from pathlib import Path
import shutil
from tdc.generation import MolGen
import rdkit
from rdkit import Chem

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--dataset",
                    type=str,
                    default="ChEMBL",
                    help="Specifies the dataset to use for creating the data. Options "
                         "are: 'ChEMBL', 'MOSES', or 'ZINC'.")
args = parser.parse_args()


def save_smiles(smi_file : str, smi_list : list) -> None:
    """Saves input list of SMILES to the specified file path."""
    smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)
    for smi in smi_list:
        try:
            mol = rdkit.Chem.MolFromSmiles(smi[0])
            if mol.GetNumAtoms() < 81:  # filter out molecules with >= 81 atoms
                save = True
                for atom in mol.GetAtoms():
                    if atom.GetFormalCharge() not in [-1, 0, +1]:  # filter out molecules with large formal charge
                        save = False
                        break
                if save:
                    smi_writer.write(mol)
        except:  # likely TypeError or AttributeError e.g. "smi[0]" is "nan"
            continue
    smi_writer.close()


if __name__ == "__main__":
    print(f"* Loading {args.dataset} dataset using the TDC.")
    data      = MolGen(name=args.dataset)
    split     = data.get_split()
    HOME      = str(Path.home())
    DATA_PATH = f"./data/{args.dataset}/"
    try:
        os.mkdir(DATA_PATH)
        print(f"-- Creating dataset at {DATA_PATH}")
    except FileExistsError:
        shutil.rmtree(DATA_PATH)
        os.mkdir(DATA_PATH)
        print(f"-- Removed old directory at {DATA_PATH}")
        print(f"-- Creating new dataset at {DATA_PATH}")

    print(f"* Re-saving {args.dataset} dataset in a format GraphINVENT can parse.")
    print("-- Saving training data...")
    save_smiles(smi_file=f"{DATA_PATH}train.smi", smi_list=split["train"].values)
    print("-- Saving testing data...")
    save_smiles(smi_file=f"{DATA_PATH}test.smi", smi_list=split["test"].values)
    print("-- Saving validation data...")
    save_smiles(smi_file=f"{DATA_PATH}valid.smi", smi_list=split["valid"].values)

    # # delete the raw downloaded files
    # dir_path = "./data/"
    # shutil.rmtree(dir_path)
    print("Done.", flush=True)
