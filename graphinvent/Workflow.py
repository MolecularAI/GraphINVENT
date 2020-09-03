# load general packages and functions
import numpy as np
import pickle
import shutil
import time
import torch
import torch.utils.tensorboard
from tqdm import tqdm
import os

# load program-specific functions
import analyze as anal
import preprocessing as prep
from BlockDatasetLoader import BlockDataLoader, HDFDataset
import generate
import loss
import models
import util

# defines `Workflow` class



# set default torch dtype
torch.set_default_dtype(torch.float32)

class Workflow:
    """ Single `Workflow` class split up into different functions for
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
    def __init__(self, constants):

        self.start_time = time.time()

        self.C = constants

        # define path variables for various datasets
        self.test_h5_path = self.C.test_set[:-3] + "h5"
        self.train_h5_path = self.C.training_set[:-3] + "h5"
        self.valid_h5_path = self.C.validation_set[:-3] + "h5"

        # create placeholders
        self.model = None
        self.optimizer = None

        self.test_dataloader = None
        self.train_dataloader = None
        self.valid_dataloader = None

        self.ts_properties = None
        self.current_epoch = None
        self.restart_epoch = None
        self.nll_per_action = None

        self.tensorboard_writer = None

    def preprocess_test_data(self):
        """ Converts test dataset to HDF file format.
        """
        print("* Preprocessing test data.", flush=True)
        prep.create_HDF_file(self.C.test_set)

        self.print_time_elapsed()

    def preprocess_train_data(self):
        """ Converts training dataset to HDF file format.
        """
        print("* Preprocessing training data.", flush=True)
        prep.create_HDF_file(self.C.training_set, is_training_set=True)

        self.print_time_elapsed()

    def preprocess_valid_data(self):
        """ Converts validation dataset to HDF file format.
        """
        print("* Preprocessing validation data.", flush=True)
        prep.create_HDF_file(self.C.validation_set)

        self.print_time_elapsed()

    def get_dataloader(self, hdf_path, data_description=None):
        """ Loads preprocessed data as `torch.utils.data.DataLoader` object.

        Args:
          data_path (str) : Path to HDF data to be read.
          data_description (str) : Used for printing status (e.g. "test data").
        """
        if data_description is None:
            data_description = "data"
        print(f"* Loading preprocessed {data_description}.", flush=True)

        dataset = HDFDataset(hdf_path)
        dataloader = BlockDataLoader(dataset=dataset,
                                     batch_size=self.C.batch_size,
                                     block_size=self.C.block_size,
                                     shuffle=True,
                                     n_workers=self.C.n_workers,
                                     pin_memory=True)
        self.print_time_elapsed()

        return dataloader

    def get_ts_properties(self):
        """ Loads the training sets properties from CSV as a dictionary, properties
        are used later for model evaluation.
        """
        filename = self.C.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)

    def define_model_and_optimizer(self):
        """ Defines the model (`self.model`) and the optimizer (`self.optimizer`).
        """
        print("* Defining model and optimizer.", flush=True)
        job_dir = self.C.job_dir

        if self.C.restart:
            print("-- Loading model from previous saved state.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            self.model = torch.load(f"{job_dir}model_restart_{self.restart_epoch}.pth")

            print(
                f"-- Backing up as "
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth.",
                flush=True,
            )
            shutil.copyfile(
                f"{job_dir}model_restart_{self.restart_epoch}.pth",
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth",
            )

        else:
            print("-- Initializing model from scratch.", flush=True)
            self.model = models.initialize_model()

            self.restart_epoch = 0

        start_epoch = self.restart_epoch + 1
        end_epoch = start_epoch + self.C.epochs

        print("-- Defining optimizer.", flush=True)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.C.init_lr,
            weight_decay=self.C.weight_decay,
        )

        return start_epoch, end_epoch

    def preprocess_phase(self):
        """ Preprocesses all the datasets (validation, training, and testing).
        """
        if not self.C.restart:  # start preprocessing job from scratch
            if (
                os.path.exists(self.valid_h5_path)
                or os.path.exists(self.test_h5_path)
                or os.path.exists(self.train_h5_path)
            ):
                raise OSError(
                    f"There currently exist(s) pre-created *.h5 file(s) in the "
                    f"dataset directory. If you would like to proceed with "
                    f"creating new ones, please delete them and rerun the "
                    f"program. Otherwise, check your input file."
                )
            self.preprocess_valid_data()
            self.preprocess_test_data()
            self.preprocess_train_data()
        else:  # restart existing preprocessing job
            # as some datasets may have already been preprocessed, check for this
            if os.path.exists(self.train_h5_path + ".chunked") or os.path.exists(self.test_h5_path):
                print(
                    f"-- Restarting preprocessing job from 'train.h5' "
                    f"(skipping over 'test.h5' and 'valid.h5' as they seem "
                    f"to be finished).",
                    flush=True,
                )
                self.preprocess_train_data()
            elif os.path.exists(self.test_h5_path + ".chunked") or os.path.exists(self.valid_h5_path):
                print(
                    f"-- Restarting preprocessing job from 'test.h5' "
                    f"(skipping over 'valid.h5' as it appears to be "
                    f"finished).",
                    flush=True,
                )
                self.preprocess_test_data()
                self.preprocess_train_data()
            elif os.path.exists(self.valid_h5_path + ".chunked"):
                print(f"-- Restarting preprocessing job from 'valid.h5'", flush=True)
                self.preprocess_valid_data()
                self.preprocess_test_data()
                self.preprocess_train_data()
            else:
                raise ValueError(
                    "Warning: Nothing to restart! Check input "
                    "file and/or submission script."
                )

    def training_phase(self):
        """ Trains model (`self.model`) and generates graphs.
        """
        self.train_dataloader = self.get_dataloader(
            hdf_path=self.train_h5_path,
            data_description="training set"
        )
        self.valid_dataloader = self.get_dataloader(
            hdf_path=self.valid_h5_path,
            data_description="validation set"
        )

        self.get_ts_properties()

        self.initialize_output_files()

        start_epoch, end_epoch = self.define_model_and_optimizer()

        print("* Beginning training.", flush=True)
        n_processed_batches = 0
        for epoch in range(start_epoch, end_epoch):

            self.current_epoch = epoch
            n_processed_batches = self.train_epoch(n_processed_batches=n_processed_batches)

            # evaluate model every `sample_every` epochs (not every epoch)
            if epoch % self.C.sample_every == 0:
                self.evaluate_model()
            else:
                util.write_model_status(score="NA")  # score not computed

        self.print_time_elapsed()

    def generation_phase(self):
        """ Generates molecules from a pre-trained model.
        """
        self.get_ts_properties()

        self.restart_epoch = self.C.generation_epoch
        print(f"* Loading model from previous saved state (Epoch {self.restart_epoch}).", flush=True)
        model_path = self.C.job_dir + f"model_restart_{self.restart_epoch}.pth"
        self.model = torch.load(model_path)

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.C.n_samples)

        self.print_time_elapsed()

    def testing_phase(self):
        """ Evaluates model using test set data.
        """
        self.test_dataloader = self.get_dataloader(self.test_h5_path, "test set")
        self.get_ts_properties()

        self.restart_epoch = util.get_restart_epoch()
        print(f"* Loading model from previous saved state (Epoch {self.restart_epoch}).", flush=True)
        self.model = torch.load(
            self.C.job_dir + f"model_restart_{self.restart_epoch}.pth"
        )

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.C.n_samples)

            print("* Evaluating model.", flush=True)
            anal.evaluate_model(valid_dataloader=self.test_dataloader,
                                train_dataloader=self.train_dataloader,
                                nll_per_action=self.nll_per_action,
                                model=self.model)

        self.print_time_elapsed()

    ####### from here on down are functions used in the various phases #######
    def evaluate_model(self):
        """ Evaluates model by calculating the UC-JSD from generated structures.
        Saves model scores in `validation.csv` and then saves model state.
        """
        self.model.eval()      # sets layers to eval mode (e.g. norm, dropout)
        with torch.no_grad():  # deactivates autograd engine

            # generate graphs required for model evaluation
            # note that evaluation of the generated graphs happens in
            # `generate_graphs()`, and molecules are saved as `self` attributes
            self.generate_graphs(n_samples=self.C.n_samples, evaluation=True)

            print("* Evaluating model.", flush=True)
            anal.evaluate_model(valid_dataloader=self.valid_dataloader,
                                train_dataloader=self.train_dataloader,
                                nll_per_action=self.nll_per_action,
                                model=self.model)

            self.nll_per_action = None  # don't need anymore

            print(f"* Saving model state at Epoch {self.current_epoch}.", flush=True)

            # `pickle.HIGHEST_PROTOCOL` good for large objects
            model_path_and_filename = (self.C.job_dir + f"model_restart_{self.current_epoch}.pth")
            torch.save(obj=self.model,
                       f=model_path_and_filename,
                       pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def initialize_output_files(self):
        """ Creates output files (with appropriate headers) for new (i.e.
        non-restart) jobs. If restart a job, and all new output will be appended
        to existing output files.
        """
        if not self.C.restart:
            print("* Touching output files.", flush=True)
            # begin writing `generation.csv` file
            csv_path_and_filename = self.C.job_dir + "generation.csv"
            util.properties_to_csv(
                prop_dict=self.ts_properties,
                csv_filename=csv_path_and_filename,
                epoch_key="Training set",
                append=False,
            )

            # begin writing `convergence.csv` file
            util.write_model_status(append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.C.job_dir + "generation/", exist_ok=True)

    def generate_graphs(self, n_samples, evaluation=False):
        """ Generates `n_graphs` molecular graphs and evaluates them. Generates
        the graphs in batches the size of `self.C.batch_size` or `n_samples` (int),
        whichever is smaller.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)

        generation_batch_size = min(self.C.batch_size, n_samples)

        n_generation_batches = int(n_samples/self.C.batch_size)

        # generate graphs in batches
        for idx in range(0, n_generation_batches + 1):
            print("Batch", idx, "of", n_generation_batches)

            # generate one batch of graphs
            # g : generated graphs (list of `GenerationGraph`s)
            # a : action NLLs (torch.Tensor)
            # f : final NLLs (torch.Tensor)
            # t : termination status (torch.Tensor)
            g, a, f, t = generate.build_graphs(model=self.model,
                                               n_graphs_to_generate=n_samples,
                                               batch_size=generation_batch_size)

            # analyze properties of new graphs and save results
            anal.evaluate_generated_graphs(generated_graphs=g,
                                           termination=t,
                                           nlls=f,
                                           start_time=self.start_time,
                                           ts_properties=self.ts_properties,
                                           generation_batch_idx=idx)

            # keep track of NLLs per action if `evaluation`==True
            # note that only NLLs for the first batch are kept, as only a few
            # are needed to evaluate the model (more efficient than saving all)
            if evaluation and idx == 0:
                self.nll_per_action = a

    def print_time_elapsed(self):
        """ Prints elapsed time since input `start_time`.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def train_epoch(self, n_processed_batches=0):
        """ Performs one training epoch.
        """
        print(f"* Training epoch {self.current_epoch}.", flush=True)
        loss_tensor = torch.zeros(len(self.train_dataloader), device="cuda")
        self.model.train()  # ensure model is in train mode

        # each batch consists of `batch_size` molecules
        # **note: "idx" == "index"
        for batch_idx, batch in tqdm(
            enumerate(self.train_dataloader), total=len(self.train_dataloader)
        ):
            n_processed_batches += 1
            batch = [b.cuda(non_blocking=True) for b in batch]
            nodes, edges, target_output = batch

            # return the output
            output = self.model(nodes, edges)

            # clear the gradients of all optimized `(torch.Tensor)`s
            self.model.zero_grad()
            self.optimizer.zero_grad()

            # compute the loss
            batch_loss = loss.graph_generation_loss(
                output=output,
                target_output=target_output,
            )

            loss_tensor[batch_idx] = batch_loss

            # backpropagate
            batch_loss.backward()
            self.optimizer.step()

            # update the learning rate
            self.update_learning_rate(n_batches=n_processed_batches)

        util.write_model_status(
            epoch=self.current_epoch,
            lr=self.optimizer.param_groups[0]["lr"],
            loss=torch.mean(loss_tensor),
        )
        return n_processed_batches

    def update_learning_rate(self, n_batches):
        """ Updates the learning rate.

        Args:
          n_batches (int) : Number of batches which have already been processed
            during training.
        """
        criterion1 = n_batches < self.C.lr_ramp_up_minibatches
        criterion2 = n_batches % (self.C.lrdi + self.C.lr_ramp_up_minibatches * self.C.ramp_up_lr) == 0

        if self.C.ramp_up_lr and criterion1:
            # calculate what the "maximum" learning rate should be given the
            # input params, and ramp up the learning rate
            max_lr = self.C.max_rel_lr * self.C.init_lr
            lr_ramp_up_factor = np.exp(np.log(max_lr / self.C.init_lr) / self.C.lr_ramp_up_minibatches)

            # learning rate will increase if not `maximum_lr` already
            util.update_lr(optimizer=self.optimizer,
                           scale_factor=lr_ramp_up_factor,
                           maximum_lr=max_lr)

        elif criterion2:
            # decreate the learning rate
            min_lr = self.C.min_rel_lr * self.C.init_lr
            util.update_lr(optimizer=self.optimizer,
                           scale_factor=self.C.lrdf**n_batches,
                           minimum_lr=min_lr)
