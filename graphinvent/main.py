# load general packages and functions
import datetime

# load GraphINVENT-specific functions
import util
from parameters.constants import constants
from Workflow import Workflow

# suppress minor warnings
util.suppress_warnings()

# defines and runs the job


def main():
    """
    Defines the type of job (preprocessing, training, generation, or testing),
    writes the job parameters (for future reference), and runs the job.
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=constants)

    job_type = constants.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters
        util.write_preprocessing_parameters(params=constants)

        # preprocess all datasets
        workflow.preprocess_phase()

    elif job_type == "train":
        # write training parameters
        util.write_job_parameters(params=constants)

        # train model and generate graphs
        workflow.training_phase()

    elif job_type == "generate":
        # write generation parameters
        util.write_job_parameters(params=constants)

        # generate molecules only
        workflow.generation_phase()

    elif job_type == "test":
        # write testing parameters
        util.write_job_parameters(params=constants)

        # evaluate best model using the test set data
        workflow.testing_phase()

    else:
        raise NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
