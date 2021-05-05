# load general packages and functions
from collections import namedtuple
import pickle
import shutil
import time
import os
from typing import Union, Tuple
import torch
import torch.utils.tensorboard
from tqdm import tqdm

# load GraphINVENT-specific functions
from Analyzer import Analyzer
from DataProcesser import DataProcesser
from BlockDatasetLoader import BlockDataLoader, HDFDataset
from Generator import GraphGenerator
import gnn.mpnn
import util

# defines `Workflow` class for carrying out specific types of jobs


class Workflow:
    """
    Single `Workflow` class for carrying out the following processes:
        1) preprocessing various molecular datasets
        2) training generative models
        3) generating molecules using pre-trained models
        4) evaluating generative models

    The preprocessing step reads a set of molecules and generates training data
    for each molecule in HDF file format, consisting of decoding routes and
    APDs. During training, the decoding routes and APDs are used to train graph
    neural network models to generate new APDs, from which actions are
    stochastically sampled and used to build new molecular graphs. During
    generation, a pre-trained model is used to generate a fixed number of
    structures. During evaluation, metrics are calculated for the test set.
    """
    def __init__(self, constants : namedtuple) -> None:

        self.start_time = time.time()

        self.constants = constants

        # define path variables for various datasets
        self.test_h5_path = self.constants.test_set[:-3] + "h5"
        self.train_h5_path = self.constants.training_set[:-3] + "h5"
        self.valid_h5_path = self.constants.validation_set[:-3] + "h5"

        # create placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.test_dataloader = None
        self.train_dataloader = None
        self.valid_dataloader = None

        self.ts_properties = None
        self.current_epoch = None
        self.restart_epoch = None
        self.analyzer = None
        self.nll_per_action = None

    def preprocess_test_data(self) -> None:
        """
        Converts test dataset to HDF file format.
        """
        print("* Preprocessing test data.", flush=True)
        test_set_preprocesser = DataProcesser(path=self.constants.test_set)
        test_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def preprocess_train_data(self) -> None:
        """
        Converts training dataset to HDF file format.
        """
        print("* Preprocessing training data.", flush=True)
        train_set_preprocesser = DataProcesser(path=self.constants.training_set,
                                               is_training_set=True)
        train_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def preprocess_valid_data(self) -> None:
        """
        Converts validation dataset to HDF file format.
        """
        print("* Preprocessing validation data.", flush=True)
        valid_set_preprocesser = DataProcesser(path=self.constants.validation_set)
        valid_set_preprocesser.preprocess()

        self.print_time_elapsed()

    def get_dataloader(self, hdf_path : str,
                       data_description : Union[str, None]=None) -> torch.utils.data.DataLoader:
        """
        Loads preprocessed data (training, validation, or test set) into a PyTorch Dataloader.

        Args:
        ----
            data_path (str) : Path to HDF data to be read.
            data_description (str) : Used for printing status (e.g. "test data").

        Returns:
        -------
            dataloader (torch.utils.data.DataLoader) : PyTorch Dataloader.
        """
        if data_description is None:
            data_description = "data"

        print(f"* Loading preprocessed {data_description}.", flush=True)

        dataset = HDFDataset(hdf_path)
        dataloader = BlockDataLoader(dataset=dataset,
                                     batch_size=self.constants.batch_size,
                                     block_size=self.constants.block_size,
                                     shuffle=True,
                                     n_workers=self.constants.n_workers,
                                     pin_memory=True)
        self.print_time_elapsed()

        return dataloader

    def load_training_set_properties(self) -> None:
        """
        Loads the training sets properties from CSV into a dictionary. The training
        set properties are used during model evaluation.
        """
        filename = self.constants.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)

    def define_model_and_optimizer(self) -> Tuple[time.time, time.time]:
        """
        Defines the model, optimizer, and scheduler.
        """
        print("* Defining model.", flush=True)
        job_dir = self.constants.job_dir

        self.model = self.create_model()

        if self.constants.restart:
            print("-- Loading model from previous saved state.", flush=True)
            self.restart_epoch = util.get_restart_epoch()

            try:
                # for loading models created using GraphINVENT v1.0 (will raise an exception
                # if model was created with GraphINVENT v2.0)
                self.model.state_dict = torch.load(f"{job_dir}model_restart_{self.restart_epoch}.pth").state_dict()
            except AttributeError:
                # for loading models created using GraphINVENT v2.0
                self.model.load_state_dict(torch.load(f"{job_dir}model_restart_{self.restart_epoch}.pth"))

            print(f"-- Backing up as {job_dir}model_restart_{self.restart_epoch}_restarted.pth.",
                  flush=True)
            shutil.copyfile(
                f"{job_dir}model_restart_{self.restart_epoch}.pth",
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth",
            )
        else:
            self.restart_epoch = 0

        start_epoch = self.restart_epoch + 1
        end_epoch = start_epoch + self.constants.epochs

        print("-- Defining optimizer.", flush=True)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.constants.init_lr)
        print("-- Defining scheduler.", flush=True)
        max_allowable_lr =  self.constants.max_rel_lr * self.constants.init_lr
        n_batches = len(self.train_dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                             max_lr=max_allowable_lr,
                                                             steps_per_epoch=n_batches,
                                                             epochs=self.constants.epochs)

        return start_epoch, end_epoch

    def create_model(self) -> torch.nn.modules:
        """
        Initializes the model to be trained. Possible models are: "MNN", "S2V",
        "AttS2V", "GGNN", "AttGGNN", or "EMN".

        Returns:
        -------
            net (SummationMPNN, AggregationMPNN or EdgeMPNN) : Neural net model.
        """
        if self.constants.model == "MNN":
            net = gnn.mpnn.MNN(constants=self.constants)
        elif self.constants.model == "S2V":
            net = gnn.mpnn.S2V(constants=self.constants)
        elif self.constants.model == "AttS2V":
            net = gnn.mpnn.AttentionS2V(constants=self.constants)
        elif self.constants.model == "GGNN":
            net = gnn.mpnn.GGNN(constants=self.constants)
        elif self.constants.model == "AttGGNN":
            net = gnn.mpnn.AttentionGGNN(constants=self.constants)
        elif self.constants.model == "EMN":
            net = gnn.mpnn.EMN(constants=self.constants)
        else:
            raise ValueError("Invalid model entered.")

        if self.constants.device == "cuda":
            net = net.to("cuda", non_blocking=True)

        return net

    def preprocess_phase(self) -> None:
        """
        Preprocesses all the datasets (validation, training, and testing).
        """
        if not self.constants.restart:
            # start preprocessing job from scratch
            hdf_files_in_data_dir = bool(os.path.exists(self.valid_h5_path)
                                         or os.path.exists(self.test_h5_path)
                                         or os.path.exists(self.train_h5_path))
            if hdf_files_in_data_dir:
                raise OSError(
                    "There currently exist(s) pre-created *.h5 file(s) in the dataset directory. "
                    "If you would like to proceed with creating new ones, please delete them and "
                    "rerun the program. Otherwise, check your input file."
                )
            self.preprocess_valid_data()
            self.preprocess_test_data()
            self.preprocess_train_data()

        else:
            # restart existing preprocessing job

            # first determine where to restart based on which HDF files have been created
            if os.path.exists(self.train_h5_path + ".chunked") or os.path.exists(self.test_h5_path):
                print(
                    "-- Restarting preprocessing job from 'train.h5' (skipping over "
                    "'test.h5' and 'valid.h5' as they seem to be finished).",
                    flush=True,
                )
                self.preprocess_train_data()
            elif os.path.exists(self.test_h5_path + ".chunked") or os.path.exists(self.valid_h5_path):
                print(
                    "-- Restarting preprocessing job from 'test.h5' (skipping over "
                    "'valid.h5' as it appears to be finished).",
                    flush=True,
                )
                self.preprocess_test_data()
                self.preprocess_train_data()
            elif os.path.exists(self.valid_h5_path + ".chunked"):
                print("-- Restarting preprocessing job from 'valid.h5'", flush=True)
                self.preprocess_valid_data()
                self.preprocess_test_data()
                self.preprocess_train_data()
            else:
                raise ValueError(
                    "Warning: Nothing to restart! Check input file and/or submission script."
                )

    def training_phase(self) -> None:
        """
        Trains model and generates graphs.
        """
        print("* Setting up training job.", flush=True)
        self.train_dataloader = self.get_dataloader(hdf_path=self.train_h5_path,
                                                    data_description="training set")
        self.valid_dataloader = self.get_dataloader(hdf_path=self.valid_h5_path,
                                                    data_description="validation set")

        self.load_training_set_properties()
        self.create_output_files()
        self.analyzer = Analyzer(valid_dataloader=self.valid_dataloader,
                                 train_dataloader=self.train_dataloader,
                                 start_time=self.start_time)

        start_epoch, end_epoch = self.define_model_and_optimizer()

        print("* Beginning training.", flush=True)
        for epoch in range(start_epoch, end_epoch):

            self.current_epoch = epoch
            avg_train_loss = self.train_epoch()
            avg_valid_loss = self.validation_epoch()

            util.write_model_status(epoch=self.current_epoch,
                                    lr=self.optimizer.param_groups[0]["lr"],
                                    training_loss=avg_train_loss,
                                    validation_loss=avg_valid_loss)

            self.evaluate_model()

        self.print_time_elapsed()

    def generation_phase(self) -> None:
        """
        Generates molecules using a pre-trained model.
        """
        print("* Setting up generation job.", flush=True)
        self.load_training_set_properties()
        self.restart_epoch = self.constants.generation_epoch
        self.analyzer = Analyzer(valid_dataloader=None,
                                 train_dataloader=None,
                                 start_time=self.start_time)

        print(f"* Loading model from saved state (Epoch {self.restart_epoch}).", flush=True)
        model_path = self.constants.job_dir + f"model_restart_{self.restart_epoch}.pth"
        self.model = self.create_model()
        try:
            # for loading models created using GraphINVENT v1.0 (will raise an exception
            # if model was created with GraphINVENT v2.0)
            self.model.state_dict = torch.load(model_path).state_dict()
        except AttributeError:
            # for loading models created using GraphINVENT v2.0
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.constants.n_samples)

        self.print_time_elapsed()

    def testing_phase(self) -> None:
        """
        Evaluates model using test set data.
        """
        self.test_dataloader = self.get_dataloader(self.test_h5_path, "test set")
        self.load_training_set_properties()
        self.restart_epoch = util.get_restart_epoch()

        print(f"* Loading model from previous saved state (Epoch {self.restart_epoch}).", flush=True)
        model_path = self.constants.job_dir + f"model_restart_{self.restart_epoch}.pth"
        self.model = self.create_model()
        try:
            # for loading models created using GraphINVENT v1.0 (will raise an exception
            # if model was created with GraphINVENT v2.0)
            self.model.state_dict = torch.load(model_path).state_dict()
        except AttributeError:
            # for loading models created using GraphINVENT v2.0
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.constants.n_samples)

            print("* Evaluating model.", flush=True)
            self.analyzer.model = self.model
            self.analyzer.evaluate_model(nll_per_action=self.nll_per_action)

        self.print_time_elapsed()

    def evaluate_model(self):
        """
        Evaluates model every `sample_every` epochs by calculating the UC-JSD from generated
        structures. Saves model scores in `validation.log` and then saves model state.
        """
        if self.current_epoch % self.constants.sample_every == 0:
            self.model.eval()      # sets layers to eval mode (e.g. norm, dropout)
            with torch.no_grad():  # deactivates autograd engine

                # generate graphs required for model evaluation (molecules are saved as
                # `self` attributes)
                self.generate_graphs(n_samples=self.constants.n_samples, evaluation=True)

                print("* Evaluating model.", flush=True)
                self.analyzer.model = self.model
                self.analyzer.evaluate_model(nll_per_action=self.nll_per_action)

                #self.nll_per_action = None  # don't need anymore

                print(f"* Saving model state at Epoch {self.current_epoch}.", flush=True)
                # `pickle.HIGHEST_PROTOCOL` good for large objects
                model_path = self.constants.job_dir + f"model_restart_{self.current_epoch}.pth"
                torch.save(obj=self.model.state_dict(),
                           f=model_path,
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)
        else:
            util.write_model_status(score="NA")  # score not computed, so use placeholder

    def create_output_files(self) -> None:
        """
        Creates output files (with appropriate headers) for new (i.e. non-restart) jobs.
        If restart a job, all new output will be appended to existing output files.
        """
        if not self.constants.restart:
            print("* Touching output files.", flush=True)
            # begin writing `generation.log` file
            csv_path_and_filename = self.constants.job_dir + "generation.log"
            util.properties_to_csv(prop_dict=self.ts_properties,
                                   csv_filename=csv_path_and_filename,
                                   epoch_key="Training set",
                                   append=False)

            # begin writing `convergence.log` file
            util.write_model_status(append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.constants.job_dir + "generation/", exist_ok=True)

    def generate_graphs(self, n_samples : int, evaluation : bool=False) -> None:
        """
        Generates molecular graphs and evaluates them. Generates the graphs in batches of either
        the size of the mini-batches or `n_samples`, whichever is smaller.

        Args:
        ----
            n_samples (int) : How many graphs to generate.
            evaluation (bool) : Indicates whether the model will be evaluated, in which
              case we will also need the NLL per action for the generated graphs.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)
        generation_batch_size = min(self.constants.batch_size, n_samples)
        n_generation_batches = int(n_samples/generation_batch_size)

        generator = GraphGenerator(model=self.model, batch_size=generation_batch_size)

        # generate graphs in batches
        for idx in range(0, n_generation_batches + 1):
            print("Batch", idx, "of", n_generation_batches)

            # generate one batch of graphs
            graphs, action_nlls, final_nlls, termination = generator.sample()

            # analyze properties of new graphs and save results
            self.analyzer.evaluate_generated_graphs(generated_graphs=graphs,
                                                    termination=termination,
                                                    nlls=final_nlls,
                                                    ts_properties=self.ts_properties,
                                                    generation_batch_idx=idx)

            # keep track of NLLs per action; note that only NLLs for the first batch are kept,
            # as only a few are needed to evaluate the model (more efficient than saving all)
            if evaluation and idx == 0:
                self.nll_per_action = action_nlls

    def print_time_elapsed(self) -> None:
        """
        Prints elapsed time since the program started running.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def train_epoch(self) -> float:
        """
        Performs one training epoch.

        Returns:
        -------
            average training loss (float)
        """
        print(f"* Training epoch {self.current_epoch}.", flush=True)
        training_loss_tensor = torch.zeros(len(self.train_dataloader),
                                           device=self.constants.device)

        self.model.train()  # ensure model is in train mode
        for batch_idx, batch in tqdm(enumerate(self.train_dataloader),
                                     total=len(self.train_dataloader)):
            if self.constants.device == "cuda":
                batch = [b.to("cuda", non_blocking=True) for b in batch]

            nodes, edges, target_output = batch
            output = self.model(nodes, edges)

            self.model.zero_grad()
            self.optimizer.zero_grad()

            batch_loss = self.loss(output=output, target_output=target_output)
            training_loss_tensor[batch_idx] = batch_loss

            # backpropagate
            batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return torch.mean(training_loss_tensor)

    def validation_epoch(self) -> float:
        """
        Performs one validation epoch.

        Returns:
        -------
            average training loss (float)
        """
        print(f"* Evaluating epoch {self.current_epoch}.", flush=True)
        validation_loss_tensor = torch.zeros(len(self.valid_dataloader),
                                             device=self.constants.device)

        self.model.eval()  # ensure model is in eval mode for computing the validation loss
        with torch.no_grad():

            for batch_idx, batch in tqdm(enumerate(self.valid_dataloader),
                                         total=len(self.valid_dataloader)):
                if self.constants.device == "cuda":
                    batch = [b.to("cuda", non_blocking=True) for b in batch]

                nodes, edges, target_output = batch
                output = self.model(nodes, edges)

                batch_loss = self.loss(output=output, target_output=target_output)
                validation_loss_tensor[batch_idx] = batch_loss

        return torch.mean(validation_loss_tensor)

    def loss(self, output : torch.Tensor, target_output : torch.Tensor) -> float:
        """
        The graph generation loss is the KL divergence between the target and predicted actions.

        Args:
        ----
            output (torch.Tensor) : Predicted APD tensor.
            target_output (torch.Tensor) : Target APD tensor.

        Returns:
        -------
            loss (float) : Average loss for this output.
        """
        # define activation function; note that one must use the softmax in the
        # KLDiv, never the sigmoid, as the distribution must sum to 1
        LogSoftmax = torch.nn.LogSoftmax(dim=1)

        output = LogSoftmax(output)

        # normalize the target output (as can contain information on > 1 graph)
        target_output = target_output/torch.sum(target_output, dim=1, keepdim=True)

        # define loss function and calculate the los
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        loss = criterion(target=target_output, input=output)

        return loss
