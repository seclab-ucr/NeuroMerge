import subprocess
import configs
import ggnn_model
import random
import modeling
import dataset
import time
import json
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score
from pathlib import Path
from docopt import docopt
import tqdm
import numpy as np
import torch
from torch_geometric.data import DataLoader  # (see below)
import sys
import os
DOC_DESC = """
Usage:
   run.py [options]

Options:
    -h --help                       Show this screen.
    --data_dir DATA_DIR             Directory(*) to of dataset. (*)=relative to repository root ProGraML/.
                                        Will overwrite the per-dataset defaults if provided.

    --log_dir LOG_DIR               Directory(*) to store logfiles and trained models relative to repository dir.
                                        [default: log/train-ggnn-model]
    --model MODEL                   The model to run.
    --dataset DATASET               The dataset to us.
    --config CONFIG                 Path(*) to a config json dump with params.
    --config_json CONFIG_JSON       Config json with params.
    --restore CHECKPOINT            Path(*) to a model file to restore from.
    --skip_restore_config           Whether to skip restoring the config from CHECKPOINT.
    --test                          Test the model without training.
    --use-class-weights             Whether to use class weights.
    --restore_by_pattern PATTERN    Restore newest model of this name from log_dir and
                                        continue training. (AULT specific!)
                                        PATTERN is a string that can be grep'ed for.
    --kfold                         Run kfold cross-validation iff kfold is set.
                                        Splits are currently dataset specific.
    --transfer MODEL                The model-class to transfer to.
                                    The args specified will be applied to the transferred model to the extend applicable, e.g.
                                        training params and Readout module specifications, but not to the transferred model trunk.
                                        However, we strongly recommend to make all trunk-parameters match, in order to be able
                                        to restore from transferred checkpoints without having to pass a matching config manually.
    --transfer_mode MODE            One of frozen, finetune (but not yet implemented) [default: frozen]
                                        Mode frozen also sets all dropout in the restored model to zero (the newly initialized
                                        readout function can have dropout nonetheless, depending on the config provided).
    --skip_save_every_epoch         Save latest model after every epoch (on a rolling basis).
"""


# make this file executable from anywhere
full_path = os.path.realpath(__file__)
REPO_ROOT = full_path.rsplit("NeuSE", maxsplit=1)[0] + "NeuSE"
ROOT_SCRIPT_DIR = REPO_ROOT + "/scripts"
sys.path.append(ROOT_SCRIPT_DIR)

from utils import visualize_loss_curve  # noqa

print("Script path: %s" % str(full_path))
print("Repository path: %s" % str(REPO_ROOT))
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, REPO_ROOT)
sys.path.append(ROOT_SCRIPT_DIR)
REPO_ROOT = Path(REPO_ROOT)


# Importing twice like this enables restoring


# Slurm gives us among others: SLURM_JOBID, SLURM_JOB_NAME,
# SLURM_JOB_DEPENDENCY (set to the value of the --dependency option)
if os.environ.get("SLURM_JOBID"):
    print("SLURM_JOB_NAME", os.environ.get("SLURM_JOB_NAME", ""))
    print("SLURM_JOBID", os.environ.get("SLURM_JOBID", ""))
    RUN_ID = "_".join(
        [os.environ.get("SLURM_JOB_NAME", ""), os.environ.get("SLURM_JOBID")]
    )
else:
    RUN_ID = str(os.getpid())


MODEL_CLASSES = {
    "ggnn_neuse": (ggnn_model.GGNNModel, configs.GGNN_NeuSE_Config),
}

DATASET_CLASSES = {  # DS, default data_dir,
    "neuse-50": (
        dataset.NeuSEDataset, os.path.abspath(
            os.path.join(str(REPO_ROOT), "dataset/merge-graph-50")
        )
    ),
    "neuse-150": (
        dataset.NeuSEDataset, os.path.abspath(
            os.path.join(str(REPO_ROOT), "dataset/merge-graph-150")
        )
    ),
    "neuse-200": (
        dataset.NeuSEDataset, os.path.abspath(
            os.path.join(str(REPO_ROOT), "dataset/merge-graph-200")
        )
    ),
    "neuse-300": (
        dataset.NeuSEDataset, os.path.abspath(
            os.path.join(str(REPO_ROOT), "dataset/merge-graph-300")
        )
    ),
}

DEBUG = False
if DEBUG:
    torch.autograd.set_detect_anomaly(True)


class Learner(object):
    def __init__(self, model, dataset, args=None, gpu_id_to_use=None, current_kfold_split=None):
        # Make class work without file being run as main
        self.args = docopt(DOC_DESC, argv=[])
        if args:
            self.args.update(args)

        # prepare logging
        self.parent_run_id = None  # for restored models
        self.run_id = f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{RUN_ID}"
        if args["--kfold"]:
            self.run_id += f"_{current_kfold_split}"

        log_dir = REPO_ROOT / self.args.get("--log_dir", ".")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"{self.run_id}_log.json"
        self.best_model_file = log_dir / f"{self.run_id}_model_best.pickle"
        self.last_model_file = log_dir / f"{self.run_id}_model_last.pickle"

        # ~~~~~~~~~~ load model ~~~~~~~~~~~~~
        if self.args.get("--restore"):
            self.model = self.restore_model(
                path=REPO_ROOT / self.args["--restore"])
        elif self.args.get("--restore_by_pattern"):
            self.model = self.restore_by_pattern(
                pattern=self.args["--restore_by_pattern"],
                log_dir=log_dir,
                current_kfold_split=current_kfold_split,
            )
        else:  # initialize fresh model
            # get model and dataset
            print("model: %s" % model)
            assert model, "Need to provide --model to initialize freshly."
            Model, Config = MODEL_CLASSES[model]

            self.global_training_step = 0
            self.current_epoch = 1

            # get config
            params = self.parse_config_params(args)
            self.config = Config.from_dict(params=params)

            test_only = self.args.get("--test", False)
            # class ratio for training
            if self.args.get('--use-class-weights', True):
                class_weights = self.get_class_weights(
                    os.path.join(DATASET_CLASSES[dataset][1], 'train')
                )
                print("Class weights for training: %s" % class_weights)
                if Model == ggnn_model.GGNNModel:
                    self.model = Model(
                        self.config, test_only=test_only, class_weights=class_weights, gpu_id_to_use=gpu_id_to_use)
            else:
                self.model = Model(
                    self.config, test_only=test_only, gpu_id_to_use=gpu_id_to_use)

        # set seeds, NB: the NN on CUDA is partially non-deterministic!
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # ~~~~~~~~~~ transfer model ~~~~~~~~
        if self.args["--transfer"] is not None:
            self.transfer_model(
                self.args["--transfer"], self.args["--transfer_mode"])

        # ~~~~~~~~~~ load data ~~~~~~~~~~~~~
        self.load_data(dataset, args["--kfold"], current_kfold_split)

        # log config to file
        config_dict = self.config.to_dict()
        with open(log_dir / f"{self.run_id}_params.json", "w") as f:
            json.dump(config_dict, f)

        # log parent run to file if run was restored
        if self.parent_run_id:
            with open(log_dir / f"{self.run_id}_parent.json", "w") as f:
                json.dump(
                    {
                        "parent": self.parent_run_id,
                        "self": self.run_id,
                        "self_config": config_dict,
                    },
                    f,
                )

        print(
            "Run %s starting with following parameters:\n%s"
            % (self.run_id, json.dumps(config_dict))
        )

    def get_class_weights(self, dataset_dir):
        fnames = os.listdir(dataset_dir)
        labels = []
        for fname in fnames:
            label = fname.replace(".pb", '').split("-")[-1]
            labels.append(label)
        class_weights = compute_class_weight(
            class_weight='balanced', classes=np.unique(labels), y=labels)
        return class_weights

    def load_data(self, dataset, kfold, current_kfold_split):
        """Set self.train_data, self.test_data, self.valid_data depending on the dataset used."""
        if not kfold:
            assert current_kfold_split is None
        if "_" in dataset:
            split = dataset.rsplit("_", maxsplit=1)[-1]
        Dataset, data_dir = DATASET_CLASSES[dataset]
        if self.args.get("--data_dir", "."):
            self.data_dir = REPO_ROOT / self.args.get("--data_dir", ".")
        else:
            self.data_dir = REPO_ROOT / data_dir

        # Switch cases by dataset
        # ~~~~~~~~~~ NeuSE on MergeGraphs ~~~~~~~~~~
        if dataset.startswith("neuse"):
            if not self.args.get("--test"):
                train_dataset = Dataset(
                    root=self.data_dir,
                    split="train",
                    train_subset=self.config.train_subset,
                    cdfg=self.config.cdfg_vocab,
                    ablation_vocab=self.config.ablation_vocab,
                )
                self.train_data = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                )

            self.valid_data = DataLoader(
                Dataset(
                    root=self.data_dir,
                    split="val",
                    cdfg=self.config.cdfg_vocab,
                    ablation_vocab=self.config.ablation_vocab,
                ),
                batch_size=self.config.batch_size * 2,
                shuffle=False,
            )
            self.test_data = DataLoader(
                Dataset(
                    root=self.data_dir,
                    split="test",
                    cdfg=self.config.cdfg_vocab,
                    ablation_vocab=self.config.ablation_vocab,
                ),
                batch_size=self.config.batch_size * 2,
                shuffle=False,
            )

    def parse_config_params(self, args):
        """Accesses self.args to parse config params from various flags."""
        params = None
        if args.get("--config"):
            with open(REPO_ROOT / args["--config"], "r") as f:
                params = json.load(f)
        elif args.get("--config_json"):
            config_string = args["--config_json"]
            # accept single quoted 'json'. This only works bc our json strings are simple enough.
            config_string = (
                config_string.replace("\\'", "'")
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
            )
            params = json.loads(config_string)
        return params

    def data2input(self, batch):
        """Glue method that converts a batch from the dataloader into the input format of the model"""
        num_graphs = batch.batch[-1].item() + 1

        edge_lists = []
        edge_positions = (
            [] if getattr(self.config, "position_embeddings", False) else None
        )

        edge_indices = list(range(3))
        if self.config.ablate_structure:
            if self.config.ablate_structure == "control":
                edge_indices[0] = -1
            elif self.config.ablate_structure == "data":
                edge_indices[1] = -1
            elif self.config.ablate_structure == "call":
                edge_indices[2] = -1
            else:
                raise ValueError("unreachable")

        for i in edge_indices:
            # mask by edge type
            mask = batch.edge_attr[:, 0] == i  # <M_i>
            edge_list = batch.edge_index[:, mask].t()  # <et, M_i>
            edge_lists.append(edge_list)

            if getattr(self.config, "position_embeddings", False):
                edge_pos = batch.edge_attr[mask, 1]  # <M_i>
                edge_positions.append(edge_pos)

        inputs = {
            "vocab_ids": batch.x[:, 0],
            "edge_lists": edge_lists,
            "pos_lists": edge_positions,
            "num_graphs": num_graphs,
            "graph_nodes_list": batch.batch,
            "node_types": batch.x[:, 1],
        }

        # maybe add labels
        if batch.y is not None:
            inputs.update(
                {"labels": batch.y, }
            )

        # add other stuff
        if hasattr(batch, "aux_in"):
            inputs.update({"aux_in": batch.aux_in.to(dtype=torch.float)})
        if hasattr(batch, "runtimes"):
            inputs.update({"runtimes": batch.runtimes.to(dtype=torch.float)})
        return inputs

    def run_epoch(self, loader, epoch_type, analysis_mode=False, print_precision_recall=False):
        """
        args:
            loader: a pytorch-geometric dataset loader,
            epoch_type: 'train' or 'eval'
        returns:
            loss, accuracy, instance_per_second
        """

        bar = tqdm.tqdm(total=len(loader.dataset), smoothing=0.01, unit="inst")
        if analysis_mode:
            saved_outputs = []

        epoch_loss, epoch_accuracy = 0, 0
        epoch_actual_rt, epoch_optimal_rt = 0, 0
        start_time = time.time()
        processed_graphs = 0
        predicted_targets = 0

        for step, batch in enumerate(loader):
            # prepare input
            # move batch to gpu and prepare input tensors:
            batch.to(self.model.dev)

            inputs = self.data2input(batch)
            num_graphs = inputs["num_graphs"]

            if getattr(self.config, "has_graph_labels", False):  # all graph models
                num_targets = num_graphs
            else:
                raise NotImplementedError(
                    "We don't have other nodewise models currently."
                )

            predicted_targets += num_targets
            processed_graphs += num_graphs

            #############
            # RUN MODEL FORWARD PASS

            # enter correct mode of model and fetch output
            if epoch_type == "train":
                self.global_training_step += 1
                if not self.model.training:
                    self.model.train()
                outputs = self.model(**inputs)
            else:  # not TRAIN
                if self.model.training:
                    self.model.eval()
                    self.model.opt.zero_grad()
                with torch.no_grad():  # don't trace computation graph!
                    outputs = self.model(**inputs)

            if analysis_mode:
                # TODO I don't know whether the outputs are properly cloned, moved to cpu and detached or not.
                saved_outputs.append(outputs)

            if hasattr(batch, "runtimes"):
                (
                    logits,
                    accuracy,
                    correct,
                    targets,
                    actual_rt,
                    optimal_rt,
                    graph_features,
                    *unroll_stats,
                ) = outputs
                epoch_actual_rt += torch.sum(actual_rt).item()
                epoch_optimal_rt += torch.sum(optimal_rt).item()
            else:
                (
                    logits,
                    accuracy,
                    correct,
                    targets,
                    graph_features,
                    *unroll_stats,
                ) = outputs
            loss = self.model.loss((logits, graph_features), targets)

            if print_precision_recall and epoch_type == "eval":
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                precision = precision_score(targets.cpu().numpy(), preds)
                recall = recall_score(targets.cpu().numpy(), preds)

            epoch_loss += loss.item() * num_targets
            epoch_accuracy += accuracy.item() * num_targets

            # update weights
            if epoch_type == "train":
                loss.backward()
                if self.model.config.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.model.config.clip_grad_norm
                    )
                self.model.opt.step()
                self.model.opt.zero_grad()

            # update bar display
            bar_loss = epoch_loss / (predicted_targets + 1e-8)
            bar_acc = epoch_accuracy / (predicted_targets + 1e-8)
            bar.set_postfix(loss=bar_loss, acc=bar_acc, ppl=np.exp(bar_loss))
            bar.update(num_graphs)

        bar.close()

        # Return epoch stats
        mean_loss = epoch_loss / predicted_targets
        mean_accuracy = epoch_accuracy / predicted_targets
        instance_per_sec = processed_graphs / (time.time() - start_time)
        epoch_perplexity = np.exp(mean_loss)

        returns = (
            mean_loss,
            mean_accuracy,
            instance_per_sec,
            epoch_perplexity,
            epoch_actual_rt,
            epoch_optimal_rt,
        )

        if analysis_mode:
            returns += (saved_outputs,)
        return returns

    def train(self):
        log_to_save = []
        total_time_start = time.time()

        # we enter training after restore
        if self.parent_run_id is not None:
            print(f"== Epoch pre-validate epoch {self.current_epoch}")
            _, valid_acc, _, ppl, _, _ = self.run_epoch(self.valid_data, "val")
            best_val_acc = np.sum(valid_acc)
            best_val_acc_epoch = self.current_epoch
            print(
                "\r\x1b[KResumed operation, initial cum. val. acc: %.5f, ppl %.5f"
                % (best_val_acc, ppl)
            )
            self.current_epoch += 1
        else:
            (best_val_acc, best_val_acc_epoch) = (0.0, 0)

        # Training loop over epochs
        target_epoch = self.current_epoch + self.config.num_epochs
        for epoch in range(self.current_epoch, target_epoch):
            print(f"== Epoch {epoch}/{target_epoch}")

            (
                train_loss,
                train_acc,
                train_speed,
                train_ppl,
                train_art,
                train_ort,
            ) = self.run_epoch(self.train_data, "train")
            print(
                "\r\x1b[K Train: loss: %.5f | acc: %s | ppl: %s | instances/sec: %.2f | runtime: %.1f opt: %.1f"
                % (
                    train_loss,
                    f"{train_acc:.5f}",
                    train_ppl,
                    train_speed,
                    train_art,
                    train_ort,
                )
            )

            (
                valid_loss,
                valid_acc,
                valid_speed,
                valid_ppl,
                valid_art,
                valid_ort,
            ) = self.run_epoch(self.valid_data, "eval")
            print(
                "\r\x1b[K Valid: loss: %.5f | acc: %s | ppl: %s | instances/sec: %.2f | runtime: %.1f opt: %.1f"
                % (
                    valid_loss,
                    f"{valid_acc:.5f}",
                    valid_ppl,
                    valid_speed,
                    valid_art,
                    valid_ort,
                )
            )

            # maybe run test epoch
            if self.test_data is not None:
                test_loss, test_acc, test_speed, test_ppl, _, _ = self.run_epoch(
                    self.test_data, "eval"
                )
                print(
                    "\r\x1b[K Test: loss: %.5f | acc: %s | ppl: %s | instances/sec: %.2f"
                    % (test_loss, f"{test_acc:.5f}", test_ppl, test_speed)
                )

            epoch_time = time.time() - total_time_start
            self.current_epoch = epoch

            log_entry = {
                "epoch": epoch,
                "time": epoch_time,
                "train_results": (
                    train_loss,
                    train_acc,
                    train_speed,
                    train_ppl,
                    train_art,
                    train_ort,
                ),
                "valid_results": (
                    valid_loss,
                    valid_acc,
                    valid_speed,
                    valid_ppl,
                    valid_art,
                    valid_ort,
                ),
            }

            if self.test_data is not None:
                log_entry.update(
                    {"test_results": (test_loss, test_acc,
                                      test_speed, test_ppl)}
                )

            log_to_save.append(log_entry)

            with open(self.log_file, "w") as f:
                json.dump(log_to_save, f, indent=4)

            # Generate the newest learning curve as well
            visualize_loss_curve(str(self.log_file))

            # TODO: sum seems redundant if only one task is trained.
            val_acc = np.sum(valid_acc)  # type: float
            if val_acc > best_val_acc:
                self.save_model(epoch, self.best_model_file)
                print(
                    "  (Best epoch so far, cum. val. acc increased to %.5f from %.5f. Saving to '%s')"
                    % (val_acc, best_val_acc, self.best_model_file)
                )
                best_val_acc = val_acc
                best_val_acc_epoch = epoch
            elif epoch - best_val_acc_epoch >= self.config.patience:
                print(
                    "Stopping training after %i epochs without improvement on validation accuracy."
                    % self.config.patience
                )
                break
            if not self.args["--skip_save_every_epoch"]:
                self.save_model(epoch, self.last_model_file)
        # save last model on finish of training
        self.save_model(epoch, self.last_model_file)

    def test(self):
        log_to_save = []
        total_time_start = time.time()

        print(f"== Epoch: Test only run.")

        (
            valid_loss,
            valid_acc,
            valid_speed,
            valid_ppl,
            valid_art,
            valid_ort,
        ) = self.run_epoch(self.valid_data, "eval")
        print(
            "\r\x1b[K Valid: loss: %.5f | acc: %s | ppl: %s | instances/sec: %.2f | runtime: %.1f opt: %.1f"
            % (
                valid_loss,
                f"{valid_acc:.5f}",
                valid_ppl,
                valid_speed,
                valid_art,
                valid_ort,
            )
        )

        if self.test_data is not None:
            test_loss, test_acc, test_speed, test_ppl, _, _ = self.run_epoch(
                self.test_data, "eval"
            )
            print(
                "\r\x1b[K Test: loss: %.5f | acc: %s | ppl: %s | instances/sec: %.2f"
                % (test_loss, f"{test_acc:.5f}", test_ppl, test_speed)
            )

        epoch_time = time.time() - total_time_start

        log_entry = {
            "epoch": "test_only",
            "time": epoch_time,
            "valid_results": (
                valid_loss,
                valid_acc,
                valid_speed,
                valid_ppl,
                valid_art,
                valid_ort,
            ),
        }
        if self.test_data is not None:
            log_entry.update(
                {"test_results": (test_loss, test_acc, test_speed, test_ppl)}
            )

        log_to_save.append(log_entry)
        with open(self.log_file, "w") as f:
            json.dump(log_to_save, f, indent=4)

    def save_model(self, epoch, path):
        checkpoint = {
            "run_id": self.run_id,
            "global_training_step": self.global_training_step,
            "epoch": epoch,
            "config": self.config.to_dict(),
            "model_name": self.model.__class__.__name__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
        }
        torch.save(checkpoint, path)

    def restore_by_pattern(self, pattern, log_dir, current_kfold_split=None):
        """This method will restore the last checkpoint of a run that is identifiable by
        the pattern <pattern>. It could restore to model_last or model_best.
        However if current_kfold_split is given, it will additionally filter for this split!
        Therefore the split should not be part of the pattern.
        """
        if current_kfold_split is not None:
            checkpoints = list(
                log_dir.glob(f"*{pattern}*_{current_kfold_split}_model_*.p*")
            )
        else:
            checkpoints = list(log_dir.glob(f"*{pattern}*_model_*.p*"))
        last_mod_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
        assert (
            last_mod_checkpoint.is_file()
        ), f"Couldn't restore by jobname: No model files matching <{pattern}> found."
        return self.restore_model(last_mod_checkpoint)

    def restore_model(self, path):
        """loads and restores a model from file."""
        checkpoint = torch.load(path)
        self.parent_run_id = checkpoint["run_id"]
        self.global_training_step = checkpoint["global_training_step"]
        self.current_epoch = checkpoint["epoch"]

        config_dict = (
            checkpoint["config"]
            if isinstance(checkpoint["config"], dict)
            else checkpoint["config"].to_dict()
        )

        if not self.args.get("--skip_restore_config"):
            # maybe zero out dropout attributes
            if (
                self.args["--transfer"] is not None
                and self.args["--transfer_mode"] == "frozen"
            ):
                for key, value in config_dict.items():
                    if "dropout" in key:
                        config_dict[key] = 0.0
                        print(
                            f"*Restoring Config* Setting {key} from {value} to 0.0 while restoring config from checkpoint for transfer."
                        )
            config = getattr(
                configs, config_dict["name"]).from_dict(config_dict)
            self.config = config
            print(
                f"*RESTORED* self.config = {config.name} from checkpoint {str(path)}."
            )
        else:
            print(f"Skipped restoring self.config from checkpoint!")
            assert (
                self.args.get("--model") is not None
            ), "Can only use --skip_restore_config if --model is given."
            # initialize config from --model and compare to skipped config from restore.
            _, Config = MODEL_CLASSES[self.args["--model"]]
            self.config = Config.from_dict(self.parse_config_params(args))
            self.config.check_equal(config_dict)

        test_only = self.args.get("--test", False)
        Model = getattr(modeling, checkpoint["model_name"])
        model = Model(self.config, test_only=test_only)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"*RESTORED* model parameters from checkpoint {str(path)}.")
        if not self.args.get(
            "--test", None
        ):  # only restore opt if needed. opt should be None o/w.
            model.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"*RESTORED* optimizer parameters from checkpoint as well.")
        return model

    def transfer_model(self, transfer_model_class, mode):
        """transfers the current model to a different model class.
        Resets global_training_step and current_epoch.

        Mode:
            frozen - only the new readout module will receive gradients.
            finetune - the whole network will receive gradients.
        """
        assert transfer_model_class in MODEL_CLASSES
        self.global_training_step = 0
        self.current_epoch = 1

        # freeze layers
        if mode == "frozen":
            for param in self.model.parameters():
                param.requires_grad = False

        # replace config
        _, Config = MODEL_CLASSES[transfer_model_class]
        params = self.parse_config_params(self.args)
        self.config = Config.from_dict(params=params)

        # replace readout
        if getattr(self.config, "has_aux_input", False) and getattr(
            self.config, "aux_use_better", False
        ):
            self.model.readout = modeling.BetterAuxiliaryReadout(self.config)
        elif getattr(self.config, "has_aux_input", False):
            self.model.readout = modeling.Readout(self.config)
            self.model.aux_readout = modeling.AuxiliaryReadout(self.config)
        else:
            assert not getattr(
                self.config, "aux_use_better", False
            ), "aux_use_better only with has_aux_input!"
            self.model.readout = modeling.Readout(self.config)

        # assign config to model
        self.model.config = self.config

        # re-setup model
        test_only = self.args.get("--test", False)
        assert (
            not test_only
        ), "Why transfer if you don't train? Here is not restoring a transferred model!!!"
        self.model.setup(self.config, test_only)
        # print info
        print(self.model)
        print(
            f"Number of trainable params in transferred model: {self.model.num_parameters()}"
        )


def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2):
    """Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).

        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output(
        'nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = [int(x.split(':')[1].replace('MiB', '').strip())
                for x in gpu_info]  # Remove garbage
    gpu_info = gpu_info[:min(max_gpus, len(gpu_info))]  # Limit to max_gpus
    # Assign free gpus to the current process
    gpus_to_use = ','.join(
        [str(i) for i, x in enumerate(gpu_info) if x < threshold_vram_usage])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    print(
        f'Available GPU(s): {gpus_to_use}' if gpus_to_use else 'No free GPUs found')
    return gpus_to_use


if __name__ == "__main__":
    args = docopt(DOC_DESC)
    print("Args for run.py: %s" % args)
    assert not (
        args["--config"] and args["--config_json"]
    ), "Can't decide which config to use!"
    if args.get("--model"):
        assert args.get("--model") in MODEL_CLASSES, f"Unknown model."
    if args.get("--dataset"):
        assert args.get("--dataset") in DATASET_CLASSES, f"Unknown dataset."

    # First, decide what GPU to use
    gpus_to_use = assign_free_gpus(threshold_vram_usage=13000, max_gpus=4)
    gpu_id_to_use = random.sample(gpus_to_use.split(','), 1)[0]

    if not args["--kfold"]:
        learner = Learner(
            model=args["--model"],
            dataset=args["--dataset"],
            args=args,
            gpu_id_to_use=gpu_id_to_use,
        )
        learner.test() if args.get("--test") else learner.train()
    else:  # kfold
        if args["--dataset"] in ["devmap_amd", "devmap_nvidia"]:
            num_splits = 10
        else:
            raise NotImplementedError(
                "kfold not implemented for this dataset.")

        for split in range(num_splits):
            print(f"#######################################")
            print(f"CURRENT SPLIT: {split} + 1/{num_splits}")
            print(f"#######################################")
            learner = Learner(
                model=args["--model"],
                dataset=args["--dataset"],
                args=args,
                current_kfold_split=split,
            )
            if len(learner.valid_data) == 0:
                print("***" * 20)
                print(
                    f"Validation Split is empty! Skipping split {split} + 1 / {num_splits}."
                )
                print("***" * 20)
            learner.test() if args.get("--test") else learner.train()
