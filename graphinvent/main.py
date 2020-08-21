# load general packages and functions
import datetime

# load program-specific functions
import util

util.suppress_warnings()
from parameters.constants import constants as C
from Workflow import Workflow

# define the generator and run it in the specified run mode



def main():
    """ Generates molecular graphs with different goals (training, generation,
    testing, or benchmarking).
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    graph_generator = Workflow(constants=C)

    job_type = C.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters
        util.write_preprocessing_parameters(params=C)

        # preprocess all datasets
        graph_generator.preprocess_phase()

    elif job_type == "train":
        # write training parameters
        util.write_job_parameters(params=C)

        # train model and generate graphs
        graph_generator.training_phase()

    elif job_type == "generate":
        # write generation parameters
        util.write_job_parameters(params=C)

        # generate molecules only
        graph_generator.generation_phase()

    elif job_type == "benchmark":
        # benchmark models using MOSES
        # TODO not integrated with MOSES, at the moment benchmarking is done by
        # generating N structures, copying the generated SMILES to the MOSES
        # dir, and running the benchmarking job according to MOSES instructions
        raise NotImplementedError

    elif job_type == "test":
        # write testing parameters
        util.write_job_parameters(params=C)

        # evaluate best model using the test set data
        graph_generator.testing_phase()

    else:
        return NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
