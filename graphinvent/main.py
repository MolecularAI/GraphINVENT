"""
Main function for running GraphINVENT jobs.

Examples:
--------
 * If you define an "input.csv" with desired job parameters in job_dir/:
   (graphinvent) ~/GraphINVENT$ python main.py --job_dir path/to/job_dir/
 * If you instead want to run your job using the submission scripts:
   (graphinvent) ~/GraphINVENT$ python submit-fine-tuning.py
"""
# load general packages and functions
import datetime

# load GraphINVENT-specific functions
import util
from parameters.constants import constants
from Workflow import Workflow

# suppress minor warnings
util.suppress_warnings()


def main():
    """
    Defines the type of job (preprocessing, training, generation, testing, or
    fine-tuning), writes the job parameters (for future reference), and runs
    the job.
    """
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # fix date/time

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

    elif job_type == "fine-tune":
        # write training parameters
        util.write_job_parameters(params=constants)

        # fine-tune the model and generate graphs
        workflow.learning_phase()

    else:
        raise NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
