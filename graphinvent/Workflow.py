"""
The `Workflow` specifies the recipes for what needs to be carried out for each
type of task.
"""
# load general packages and functions
from collections import namedtuple
import pickle
from copy import deepcopy
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
from GraphGenerator import GraphGenerator
from GraphGeneratorRL import GraphGeneratorRL
from ScoringFunction import ScoringFunction
import gnn.mpnn
import util


class Workflow:
    """
    Single `Workflow` class for carrying out the following processes:
        1) preprocessing various molecular datasets
        2) training generative models
        3) generating molecules using pre-trained models
        4) evaluating generative models
        5) fine-tuning generative models (via RL)

    The preprocessing step reads a set of molecules and generates training data
    for each molecule in HDF file format, consisting of decoding routes and APDs.
    During training, the decoding routes and APDs are used to train graph neural
    network models to generate new APDs, from which actions are stochastically
    sampled and used to build new molecular graphs. During generation, a
    pre-trained model is used to generate a fixed number of structures. During
    evaluation, metrics are calculated for the test set.
    """
    def __init__(self, constants : namedtuple) -> None:

        self.start_time = time.time()
        self.constants  = constants

        # define path variables for various datasets
        self.test_h5_path  = self.constants.test_set[:-3] + "h5"
        self.train_h5_path = self.constants.training_set[:-3] + "h5"
        self.valid_h5_path = self.constants.validation_set[:-3] + "h5"

        self.test_smi_path  = self.constants.test_set
        self.train_smi_path = self.constants.training_set
        self.valid_smi_path = self.constants.validation_set

        # general paramters (placeholders)
        self.optimizer     = None
        self.scheduler     = None
        self.analyzer      = None
        self.current_epoch = None
        self.restart_epoch = None

        # non-reinforcement learning parameters (placeholders)
        self.model                 = None
        self.ts_properties         = None
        self.test_dataloader       = None
        self.train_dataloader      = None
        self.valid_dataloader      = None
        self.likelihood_per_action = None

        # reinforcement learning parameters (placeholders)
        self.agent_model      = None
        self.prior_model      = None
        self.basf_model       = None  # basf stands for "best agent so far"
        self.best_avg_score   = 0.0
        self.rl_step          = 0.0
        self.scoring_function = None

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
                       data_description : Union[str, None]=None) -> \
                       torch.utils.data.DataLoader:
        """
        Loads preprocessed data (training, validation, or test set) into a
        PyTorch Dataloader.

        Args:
        ----
            data_path (str)        : Path to HDF data to be read.
            data_description (str) : Used for printing status (e.g. "test data").

        Returns:
        -------
            dataloader (torch.utils.data.DataLoader) : PyTorch Dataloader.
        """
        if data_description is None:
            data_description = "data"

        print(f"* Loading preprocessed {data_description}.", flush=True)
        dataset    = HDFDataset(hdf_path)
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
        Loads the training sets properties from CSV into a dictionary. The
        training set properties are used during model evaluation.
        """
        filename           = self.constants.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)

    def define_model_and_optimizer(self) -> Tuple[int, int]:
        """
        Defines the model, optimizer, and scheduler, depending on the type of
        job, i.e., a regular training job, a restart job, or a fine-tuning job.

        Returns:
        -------
            start_epoch (int) : Epoch at which to start training.
            end_epoch (int)   : Epoch at which to end training.
        """

        job_dir = self.constants.job_dir

        if self.constants.job_type == "fine-tune":

            print("* Defining models.", flush=True)
            self.agent_model = self.create_model()
            self.prior_model = self.create_model()
            self.basf_model  = self.create_model()

            print("-- Loading pre-trained model from previous saved state.",
                  flush=True)

            self.restart_epoch = util.get_restart_epoch()
            model_dir          = self.constants.pretrained_model_dir

            try:
                self.agent_model = util.load_saved_model(
                    model=self.agent_model,
                    path=f"{model_dir}model_restart_{self.restart_epoch}.pth"
                )
            except FileNotFoundError:
                self.agent_model = util.load_saved_model(
                    model=self.agent_model,
                    path=f"{self.constants.dataset_dir}pretrained_model.pth"
                )
            self.prior_model = deepcopy(self.agent_model)
            self.basf_model  = deepcopy(self.agent_model)

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(params=self.agent_model.parameters(),
                                              lr=self.constants.init_lr)

            start_epoch    = self.restart_epoch + 1
            end_epoch      = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr= self.constants.max_rel_lr * self.constants.init_lr,
                div_factor= 1. / self.constants.max_rel_lr,
                final_div_factor = 1. / self.constants.min_rel_lr,
                pct_start = 0.05,
                total_steps=self.constants.epochs,
                epochs=self.constants.epochs
            )

        elif self.constants.restart:

            print("* Defining model.", flush=True)
            self.model = self.create_model()

            print("-- Loading model from previous saved state.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            self.model         = util.load_saved_model(
                model=self.model,
                path=f"{job_dir}model_restart_{self.restart_epoch}.pth"
            )

            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.constants.init_lr)

            start_epoch    = self.restart_epoch + 1
            end_epoch      = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            max_allowable_lr = (
                self.constants.max_rel_lr * self.constants.init_lr
            )
            n_batches = len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_allowable_lr,
                steps_per_epoch=n_batches,
                epochs=self.constants.epochs
            )
        else:
            self.restart_epoch = 0

            print("* Defining model.", flush=True)
            self.model = self.create_model()
            
            print("-- Defining optimizer.", flush=True)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              lr=self.constants.init_lr)

            start_epoch = self.restart_epoch + 1
            end_epoch   = start_epoch + self.constants.epochs

            print("-- Defining scheduler.", flush=True)
            max_allowable_lr =  (
                self.constants.max_rel_lr * self.constants.init_lr
            )
            n_batches = len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_allowable_lr,
                steps_per_epoch=n_batches,
                epochs=self.constants.epochs
            )

        return start_epoch, end_epoch

    def create_model(self) -> torch.nn.Module:
        """
        Initializes the model to be trained. Possible models are: "MNN", "S2V",
        "AttS2V", "GGNN", "AttGGNN", or "EMN".

        Returns:
        -------
            net (torch.nn.Module) : Neural net model.
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
                    "There currently exist(s) pre-created *.h5 file(s) in the "
                    "dataset directory. If you would like to proceed with "
                    "creating new ones, please delete them and rerun the "
                    "program. Otherwise, check your input file."
                )
            if os.path.exists(self.valid_smi_path):       
                self.preprocess_valid_data()
            if os.path.exists(self.test_smi_path):       
                self.preprocess_test_data()
            if os.path.exists(self.train_smi_path):       
                self.preprocess_train_data()

        else:  # restart existing preprocessing job

            # first determine where to restart based on which HDF files have been created
            if (os.path.exists(self.train_h5_path + ".chunked") or
                os.path.exists(self.test_h5_path)):
                print(
                    "-- Restarting preprocessing job from 'train.h5' (skipping "
                    "over 'test.h5' and 'valid.h5' as they seem to be finished).",
                    flush=True,
                )
                if os.path.exists(self.train_smi_path):       
                    self.preprocess_train_data()
            elif (os.path.exists(self.test_h5_path + ".chunked") or
                  os.path.exists(self.valid_h5_path)):
                print(
                    "-- Restarting preprocessing job from 'test.h5' (skipping "
                    "over 'valid.h5' as it appears to be finished).",
                    flush=True,
                ) 
                if os.path.exists(self.test_smi_path):       
                    self.preprocess_test_data()
                if os.path.exists(self.train_smi_path):       
                    self.preprocess_train_data()
            elif os.path.exists(self.valid_h5_path + ".chunked"):
                print("-- Restarting preprocessing job from 'valid.h5'",
                      flush=True)
                if os.path.exists(self.valid_smi_path):       
                    self.preprocess_valid_data()
                if os.path.exists(self.test_smi_path):       
                    self.preprocess_test_data()
                if os.path.exists(self.train_smi_path):       
                    self.preprocess_train_data()
            else:
                raise ValueError(
                    "Warning: Nothing to restart! Check input file and/or "
                    "submission script."
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

            util.write_training_status(epoch=self.current_epoch,
                                       lr=self.optimizer.param_groups[0]["lr"],
                                       training_loss=avg_train_loss,
                                       validation_loss=avg_valid_loss)

            _ = self.evaluate_model(model_to_evaluate=self.model)

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

        print(f"* Loading model from saved state (Epoch {self.restart_epoch}).",
              flush=True)
        model_path = (f"{self.constants.job_dir}"
                      f"model_restart_{self.restart_epoch}.pth")
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.constants.n_samples)

        self.print_time_elapsed()

    def testing_phase(self) -> None:
        """
        Evaluates model using test set data.
        """
        self.test_dataloader = self.get_dataloader(self.test_h5_path,
                                                   "test set")
        self.load_training_set_properties()
        self.restart_epoch = util.get_restart_epoch()

        print(f"* Loading model from previous saved state (Epoch "
              f"{self.restart_epoch}).", flush=True)
        model_path = (f"{self.constants.job_dir}"
                      f"model_restart_{self.restart_epoch}.pth")
        self.model = self.create_model()
        self.model = util.load_saved_model(model=self.model, path=model_path)

        self.model.eval()
        with torch.no_grad():
            self.generate_graphs(n_samples=self.constants.n_samples)

            print("* Evaluating model.", flush=True)
            self.analyzer.model = self.model
            self.analyzer.evaluate_model(
                likelihood_per_action=self.likelihood_per_action
            )

        self.print_time_elapsed()

    def evaluate_model(self, model_to_evaluate : torch.nn.Module,
                       label : str="") -> Union[float, None]:
        """
        Evaluates model.

        For regular training jobs, evaluates the model every `sample_every`
        epochs by calculating the UC-JSD from generated structures. Saves model
        scores in `validation.log` and then saves model state.

        For fine-tuning jobs, evaluates the model every time function is called
        (i.e., every fine-tuning step) by computing the score of molecules
        generated by the specified model.

        Args:
        ----
            model_to_evaluate (torch.nn.Module) : Specific model to evaluate
                                                  (e.g. SummationMPNN).
            label (str)                         : Label to use for saving generated
                                                  structures from a specific graph
                                                  generation step.

        Returns:
        -------
            score (float) : Model score for fine-tuning job, otherwise simply None.
        """
        if self.constants.job_type == "fine-tune":
            model_to_evaluate.eval()  # sets layers to eval mode (e.g. norm, dropout)
            with torch.no_grad():     # deactivates autograd engine

                _, score = self.generate_graphs_rl(
                    model_a=model_to_evaluate,
                    model_b=self.prior_model,
                    is_agent=True,
                    model_a_label=label
                )

                print(f"* Saving model state at Epoch {self.current_epoch}.",
                      flush=True)
                # `pickle.HIGHEST_PROTOCOL` good for large objects
                model_path = (f"{self.constants.job_dir}"
                              f"model_restart_{self.current_epoch}.pth")
                torch.save(obj=model_to_evaluate.state_dict(),
                           f=model_path,
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

        elif self.current_epoch % self.constants.sample_every == 0:
            model_to_evaluate.eval()  # sets layers to eval mode (e.g. norm, dropout)
            with torch.no_grad():     # deactivates autograd engine
                self.generate_graphs(n_samples=self.constants.n_samples,
                                     evaluation=True)
                score = None

                print(f"* Saving model state at Epoch {self.current_epoch}.",
                      flush=True)
                # `pickle.HIGHEST_PROTOCOL` good for large objects
                model_path = (f"{self.constants.job_dir}"
                              f"model_restart_{self.current_epoch}.pth")
                torch.save(obj=model_to_evaluate.state_dict(),
                           f=model_path,
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

                print("* Evaluating model.", flush=True)
                self.analyzer.model = model_to_evaluate
                self.analyzer.evaluate_model(
                    likelihood_per_action=self.likelihood_per_action
                )

        else:
            # score not computer, so use placeholder
            util.write_training_status(score="NA")
            score = None

        return score

    def learning_phase(self) -> None:
        """
        Fine-tunes model (`self.prior_model`) via policy gradient reinforcement
        learning (`self.agent_model`).
        """
        print("* Setting up RL fine-tuning job.", flush=True)
        self.load_training_set_properties()
        self.create_output_files()

        self.analyzer = Analyzer(
            valid_dataloader=None,
            train_dataloader=None,
            start_time=self.start_time
        )

        # define the scoring function to be used
        self.scoring_function = ScoringFunction(constants=self.constants)

        start_step, end_step = self.define_model_and_optimizer()

        # evaluate model before fine-tuning
        score = self.evaluate_model(
            model_to_evaluate=self.agent_model,
            label="pre-fine-tuning"
        )

        # save the score to the analyzer
        self.analyzer.save_metrics(step=start_step, score=score, append=False)

        print("* Begin learning.", flush=True)

        for step in range(start_step, end_step):

            self.current_epoch = step
            self.learning_step()

            # evaluate model every `sample_every` epochs (not every epoch)
            if step % self.constants.sample_every == 0:
                score = self.evaluate_model(model_to_evaluate=self.agent_model,
                                            label="eval")

                # save the score to the analyzer
                self.analyzer.save_metrics(step=step, score=score)

                # check if agent's score is better than best score so far
                if score > self.best_avg_score:
                    self.best_avg_score = score

                    # update the best agent so far ("basf")
                    self.basf_model = deepcopy(self.agent_model)
                    print("-- Updated best model.", flush=True)

        self.print_time_elapsed()

    def learning_step(self) -> None:
        """
        Performs one fine-tuning step.
        """
        print(f"* Learning step {self.current_epoch}.", flush=True)
        self.agent_model.train()  # ensure model is in train mode
        self.agent_model.zero_grad()
        self.optimizer.zero_grad()

        self.prior_model.eval()
        self.basf_model.eval()

        # genereate molecules with agent model
        loss_a, score_a = self.generate_graphs_rl(model_a=self.agent_model,
                                                  model_b=self.prior_model,
                                                  is_agent=True,
                                                  model_a_label="agent")

        # generate molecules with best agent so far ("basf")
        loss_b, _ = self.generate_graphs_rl(model_a=self.basf_model,
                                            model_b=self.agent_model,
                                            is_agent=True,
                                            model_a_label="BASF")

        loss = (
            (1 - self.constants.alpha) * loss_a + self.constants.alpha * loss_b
        )

        # backpropagate
        loss.backward()
        self.optimizer.step()

        # update the learning rate
        self.scheduler.step()

        util.write_training_status(
            epoch=self.current_epoch,
            lr=self.optimizer.param_groups[0]["lr"],
            training_loss=torch.clone(loss),
            validation_loss=0.0,  # placeholder, not meaningful during fine-tuning
            score=torch.mean(torch.clone(score_a)).item()
        )

        self.rl_step += 1

    def create_output_files(self) -> None:
        """
        Creates output files (with appropriate headers) for new (i.e.
        non-restart) jobs. If restart a job, all new output will be appended
        to existing output files.
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
            util.write_training_status(append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.constants.job_dir + "generation/", exist_ok=True)

    def generate_graphs(self, n_samples : int, evaluation : bool=False) -> None:
        """
        Generates molecular graphs and evaluates them. Generates the graphs in
        batches of either the size of the mini-batches or `n_samples`, whichever
        is smaller.

        Args:
        ----
            n_samples (int)   : How many graphs to generate.
            evaluation (bool) : Indicates whether the model will be evaluated,
                                in which case we will also need the NLL per
                                action for the generated graphs.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)
        generation_batch_size = min(self.constants.batch_size, n_samples)
        n_generation_batches  = int(n_samples/generation_batch_size)

        generator = GraphGenerator(model=self.model,
                                   batch_size=generation_batch_size)

        # generate graphs in batches
        for idx in range(0, n_generation_batches + 1):
            print("Batch", idx, "of", n_generation_batches)

            # generate one batch of graphs
            (graphs, action_likelihoods, final_loglikelihoods,
             termination) = generator.sample()

            # analyze properties of new graphs and save results
            self.analyzer.evaluate_generated_graphs(
                generated_graphs=graphs,
                termination=termination,
                loglikelihoods=final_loglikelihoods,
                ts_properties=self.ts_properties,
                generation_batch_idx=idx
            )

            # keep track of NLLs per action; note that only NLLs for the first
            # batch are kept, as only a few are needed to evaluate the model
            # (more efficient than saving all)
            if evaluation and idx == 0:
                self.likelihood_per_action = action_likelihoods

    def generate_graphs_rl(self, model_a : torch.nn.Module,
                           model_b : torch.nn.Module, is_agent : bool=False,
                           model_a_label : str="") -> \
                           Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates molecular graphs during fine-tuning using two different models;
        these can be any pair of models, including the "agent", the "prior", or
        the "best-agent-so-far". The generated structures are then evaluated in
        terms of their validity and uniqueness, for use in the loss function
        (loss is not updated for invalid/duplicate molecules).

        Args:
        ----
            model_a (torch.nn.Module) : The first model, which is used for
                                        generating new molecular graphs.
            model_b (torch.nn.Module) : The second model, which is used only to
                                        compute the likelihood.
            is_agent (bool)           : Indicates if `model_a` is the agent model.
            model_a_label (str)       : Label to use when saving structures generated
                                        from `model_a`.

        Returns:
        -------
            loss_component (torch.Tensor) : Contribution to the loss from the
                                            sampled molecules.
            torch.Tensor : Average score for the sampled molecules.
        """
        n_samples = self.constants.batch_size  # how many molecules to generate

        print(f"* Generating {n_samples} molecules.", flush=True)
        if model_a_label != "":
            print(f"-- Model: {model_a_label}", flush=True)

        generator = GraphGeneratorRL(model=self.model,
                                     batch_size=self.constants.batch_size)

        # generate one batch of graphs using `model_a`
        (graphs, model_a_loglikelihoods, model_b_loglikelihoods,
         termination) = generator.sample(agent_model=model_a,
                                         prior_model=model_b)

        # analyze properties of new graphs and save results
        validity, uniqueness = self.analyzer.evaluate_generated_graphs_rl(
            generated_graphs=graphs,
            termination=termination,
            agent_loglikelihoods=model_a_loglikelihoods,
            prior_loglikelihoods=model_b_loglikelihoods,
            ts_properties=self.ts_properties,
            step=self.current_epoch,
            is_agent=is_agent,
            label=model_a_label
        )

        scores = self.scoring_function.compute_score(graphs=graphs,
                                                    termination=termination,
                                                    validity=validity,
                                                    uniqueness=uniqueness)

        if is_agent:
            util.tbwrite_loglikelihoods(
                step=self.current_epoch,
                agent_loglikelihoods=-torch.clone(model_a_loglikelihoods),
                prior_loglikelihoods=-torch.clone(model_b_loglikelihoods)
            )
        else:
            uniqueness = torch.where(scores > self.best_avg_score,
                                     uniqueness,
                                     torch.zeros(len(scores),
                                     device=self.constants.device))

        loss_component = torch.mean(
            self.compute_loss_component(
                scores=scores,
                agent_loglikelihoods=model_a_loglikelihoods,
                prior_loglikelihoods=model_b_loglikelihoods,
                uniqueness=uniqueness)
        )

        return loss_component, torch.mean(scores)

    def print_time_elapsed(self) -> None:
        """
        Prints elapsed time since the program started running.
        """
        stop_time    = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def train_epoch(self) -> float:
        """
        Performs one training epoch.

        Returns:
        -------
            torch.Tensor : Average training loss.
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

    def validation_epoch(self) -> torch.Tensor:
        """
        Performs one validation epoch.

        Args:
        ----
            return_batch (bool) : If True, returns the validation loss
                                  tensor for the entire batch.

        Returns:
        -------
            torch.Tensor : Average validation loss.
        """
        print(f"* Evaluating epoch {self.current_epoch}.", flush=True)
        validation_loss_tensor = torch.zeros(len(self.valid_dataloader),
                                             device=self.constants.device)

        self.model.eval()
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

    def loss(self, output : torch.Tensor, target_output : torch.Tensor) -> \
        torch.Tensor:
        """
        The graph generation loss is the KL divergence between the target and
        predicted actions.

        Args:
        ----
            output (torch.Tensor)        : Predicted APD tensor.
            target_output (torch.Tensor) : Target APD tensor.

        Returns:
        -------
            loss (torch.Tensor) : Average loss for this output.
        """
        # define activation function; note that one must use the softmax in the
        # KLDiv, never the sigmoid, as the distribution must sum to 1
        LogSoftmax = torch.nn.LogSoftmax(dim=1)
        output     = LogSoftmax(output)

        # normalize the target output (as can contain information on > 1 graph)
        target_output = target_output/torch.sum(target_output, dim=1, keepdim=True)

        # define loss function and calculate the los
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        loss      = criterion(target=target_output, input=output)

        return loss

    def compute_loss_component(self, scores : torch.Tensor,
                               agent_loglikelihoods : torch.Tensor,
                               prior_loglikelihoods : torch.Tensor,
                               uniqueness : torch.Tensor) -> \
                               torch.Tensor:
        """
        Computes the contributions to the loss from the log-likelihoods/scores
        of the two input models.

        Args:
        ----
            scores (torch.Tensor)               : Scores for sampled molecules based
                                                  on the user-defined scoring function.
            agent_loglikelihoods (torch.Tensor) : Log-likelihoods of generating
                                                  the sampled structures using the
                                                  agent model.
            prior_loglikelihoods (torch.Tensor) : Log-likelihoods of generating
                                                  the sampled structures using the
                                                  prior model.
            uniqueness (torch.Tensor)           : Vector specifying the uniqueness
                                                  of each sampled structure (1 -->
                                                  unique, 0 --> duplicate).

        Returns:
        -------
            torch.Tensor: The loss contributions from the input log-likelihoods.
        """
        augmented_prior_loglikelihoods = (
            prior_loglikelihoods + self.constants.sigma * scores
        )

        difference = agent_loglikelihoods - augmented_prior_loglikelihoods
        loss       = difference * difference
        mask       = (uniqueness != 0).int()
        loss       = loss * mask

        return loss
