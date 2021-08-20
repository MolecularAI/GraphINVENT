"""
Defines `ArgumentParser` for specifying job directory using command-line.
"""# load general packages and functions
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)
parser.add_argument("--job-dir",
                    type=str,
                    default="../output/",
                    help="Directory in which to write all output.")


args = parser.parse_args()

args_dict = vars(args)
job_dir = args_dict["job_dir"]
