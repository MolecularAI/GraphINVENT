# load general packages and functions
import datetime

# load program-specific functions
import util

util.suppress_warnings()
from parameters.constants import constants as C
from Workflow import Workflow

# defines and runs the job



def main():
    """ Defines the type of job (preprocessing, training, generation, or testing), 
    runs it, and writes the job parameters used.
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=C)

    job_type = C.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters
        util.write_preprocessing_parameters(params=C)

        # preprocess all datasets
        workflow.preprocess_phase()

    elif job_type == "train":
        # write training parameters
        util.write_job_parameters(params=C)

        # train model and generate graphs
        workflow.training_phase()

    elif job_type == "generate":
        # write generation parameters
        util.write_job_parameters(params=C)

        # generate molecules only
        workflow.generation_phase()

    elif job_type == "benchmark":
        # TODO not integrated with MOSES, at the moment benchmarking is done by
        # generating N structures, copying the generated SMILES to the MOSES
        # dir, and running the benchmarking job according to MOSES instructions
        raise NotImplementedError

    elif job_type == "test":
        # write testing parameters
        util.write_job_parameters(params=C)

        # evaluate best model using the test set data
        workflow.testing_phase()

    else:
        return NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
