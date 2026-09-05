"""
Module for text classification default handler
DOES NOT SUPPORT BATCH!
"""
import logging
import torch
import torch.nn.functional as F
from .base_handler import BaseHandler
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict
from programl.proto.program_graph_pb2 import ProgramGraph
from docopt import docopt
import json
import sys
import os
import getpass
from ..context import Context
import time

REPO_ROOT = "/data/" + getpass.getuser() + "/NeuSE"
print("REPO_ROOT: %s" % REPO_ROOT)
PATH_TO_PROGRAML_GGNN = REPO_ROOT + "/scripts/ggnn"
sys.path.append(PATH_TO_PROGRAML_GGNN)
PATH_TO_PROGRAML_SCRIPTS = REPO_ROOT + "/scripts"
sys.path.append(PATH_TO_PROGRAML_SCRIPTS)
PATH_TO_PROGRAML_SCRIPTS = REPO_ROOT + "/serve"
sys.path.append(PATH_TO_PROGRAML_SCRIPTS)
VOCAB_PATH = REPO_ROOT + "/dataset/vocab/programl.csv"
REPO_ROOT = Path(REPO_ROOT)

print("All searchable dirs (for imports from ggnn_classifier): %s" % sys.path)

import dataset  # noqa
from dataset import AblationVocab  # noqa
import utils  # noqa
import ggnn_model  # noqa
import configs  # noqa
from run import DOC_DESC  # noqa

logger = logging.getLogger(__name__)


def convert_merge_gragh_to_data(
    graph: ProgramGraph,
    vocabulary: Dict[str, int],
    ignore_profile_info=True,
    ablate_vocab=AblationVocab.NONE,
):
    """Converts a program graph protocol buffer to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        graph           A program graph protocol buffer.
        vocabulary      A map from node text to vocabulary indices.
        y_feature_name  The name of the graph-level feature to use as class label.
        ablate_vocab    Whether to use an ablation vocabulary.
    """
    # collect edge_index
    edge_tuples = [(edge.source, edge.target) for edge in graph.edge]
    edge_index = torch.tensor(edge_tuples).t().contiguous()

    # collect edge_attr
    positions = torch.tensor([edge.position for edge in graph.edge])
    flows = torch.tensor([int(edge.flow) for edge in graph.edge])

    edge_attr = torch.cat([flows, positions]).view(2, -1).t().contiguous()

    # collect x
    if ablate_vocab == AblationVocab.NONE:
        vocabulary_indices = vocab_ids = [
            vocabulary.get(node.text, len(vocabulary)) for node in graph.node
        ]
    elif ablate_vocab == AblationVocab.NO_VOCAB:
        vocabulary_indices = [0] * len(graph.node)
    elif ablate_vocab == AblationVocab.NODE_TYPE_ONLY:
        vocabulary_indices = [int(node.type) for node in graph.node]
    else:
        raise NotImplementedError("unreachable")

    xs = torch.tensor(vocabulary_indices)
    types = torch.tensor([int(node.type) for node in graph.node])

    x = torch.cat([xs, types]).view(2, -1).t().contiguous()

    assert (
        edge_attr.size()[0] == edge_index.size()[1]
    ), f"edge_attr={edge_attr.size()} size mismatch with edge_index={edge_index.size()}"

    data_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    # branch prediction / profile info specific
    if not ignore_profile_info:
        raise NotImplementedError(
            "profile info is not supported with the new nx2data (from programgraph) adaptation."
        )

    # make Data
    data = Data(**data_dict)

    return data

# Note that we don't have secondary abstraction layer for this handler
# at the moment. We will try to add it in the future for different
# concrete tasks (e.g., classification, regression, etc.).


class GGNNClassifier(BaseHandler):
    """
    Handler for graph classifications over MergeGraph.
    """

    def __init__(self):
        super().__init__()
        self.initialized = None
        self.args = docopt(doc=DOC_DESC, read_argv=False)

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

    def _load_pickled_model(
        self,
        model_dir,
        model_file,
        model_pt_path
    ):
        """
        Loads the model from the context.
        """
        """loads and restores a model from file."""
        dev = (
            torch.device("cuda:%s" % self.gpu_id) if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        model_path = model_pt_path
        checkpoint = torch.load(model_path, map_location=dev)
        self.parent_run_id = checkpoint["run_id"]
        self.global_training_step = checkpoint["global_training_step"]
        self.current_epoch = checkpoint["epoch"]

        config_dict = (
            checkpoint["config"]
            if isinstance(checkpoint["config"], dict)
            else checkpoint["config"].to_dict()
        )

        MODEL_CLASSES = {
            "ggnn_neuse": (ggnn_model.GGNNModel, configs.GGNN_NeuSE_Config),
        }

        self.args["--model"] = "ggnn_neuse"
        self.args["--dataset"] = "neuse"

        logging.info("model: %s" % self.args.get("--model"))

        assert (
            self.args.get("--model") is not None
        ), "Can only use --skip_restore_config if --model is given."
        # initialize config from --model and compare to skipped config from restore.

        _, Config = MODEL_CLASSES[self.args["--model"]]
        self.config = Config.from_dict(self.parse_config_params(self.args))
        self.config.check_equal(config_dict)

        test_only = self.args.get("--test", False)
        Model = getattr(ggnn_model, checkpoint["model_name"])

        model = Model(self.config, test_only=test_only)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(
            f"*RESTORED* model parameters from checkpoint {str(model_path)}.")
        if not self.args.get(
            "--test", None
        ):  # only restore opt if needed. opt should be None o/w.
            model.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"*RESTORED* optimizer parameters from checkpoint as well.")
        return model.to(dev)

    def _load_programl_graphs(self, context):
        """
        Loads the MergeGraphs from the context.
        """
        # Load the MergeGraphs.
        programl_graph_dir = REPO_ROOT / "dataset/ir-programl"
        return utils.scan_programl_graph_dir(programl_graph_dir)

    def preprocess(self, data):
        # Let us see what happens here.
        assert len(data) == 1, "[ERROR] Multiple graphs in data."
        data_input = data[0]["body"].decode()
        return data_input

    def postprocess(self, data):
        logits = data[0]
        pred = F.softmax(logits, dim=1).tolist()
        preds = [
            {
                "SHOULD_NOT_MERGE": pred[0][0],
                "SHOULD_MERGE": pred[0][1],
            }
        ]
        return preds

    def initialize(self, context):
        """
        Loads the model as well as ProGraML graphs to memory to avoid
        on-the-fly graph loading in the inference loop.
        """
        print("Initializing the GGNN model inside initialize...")
        self.initialized = False
        self.gpu_id = context._system_properties["gpu_id"]
        super().initialize(context)
        # First, load stuff.
        self.program_graphs = self._load_programl_graphs(context)
        self.vocab = dataset.load_vocabulary(Path(VOCAB_PATH))
        self.initialized = True

    # TODO: add support for batching.
    def inference(self, data, debug=False, *args, **kwargs):
        """The Inference Request is made through this function and the user
        needs to override the inference function to customize it.

        Args:
            data: [merge_record, program_name]

        Returns:
            prediction: [prediction]
        """
        merge_record = utils.parse_merge_json_from_str(data)
        program_name = merge_record.ir_fname.split("/")[-1].replace(".bc", "")
        programl_graph = self.program_graphs[program_name][0]
        merge_graph = utils.gen_merge_graph(
            merge_record, programl_graph, utils.MergeGraphVarInclusion.ONLY_MERGED)
        data = dataset.nx2data4serve(
            graph=merge_graph,
            vocabulary=self.vocab,
        )
        inputs = dataset.data2input4serve(data, self.config, dev=self.gpu_id)

        if debug:
            print("[DEBUG] Device of vocab_ids: %s" %
                  inputs["vocab_ids"].get_device())
            print("[DEBUG] Device of labels: %s" %
                  inputs["labels"].get_device())
            print("[DEBUG] Device of graph_nodes_list: %s" %
                  inputs["graph_nodes_list"].get_device())
            print(inputs["pos_lists"])
            print("Specified device: %s" % self.gpu_id)

        prediction = self.model(
            vocab_ids=inputs["vocab_ids"],
            labels=inputs["labels"],
            edge_lists=inputs["edge_lists"],
            pos_lists=inputs["pos_lists"],
            node_types=inputs["node_types"],
            graph_nodes_list=inputs["graph_nodes_list"],
            num_graphs=inputs["num_graphs"],
        )
        return prediction


if __name__ == "__main__":
    # Test #1: Load model
    # Let us make sure functions work as expected.
    ggnn = GGNNClassifier()
    model_path = REPO_ROOT / "log/train-ggnn-model/"
    ggnn._load_pickled_model(
        model_dir=model_path,
        model_file="",
        model_pt_path=model_path / "2022-01-29_22:56:55_296615_model_best.pickle"
    )

    # Test #2: Load a data sample to get inference.
    test_data_json = REPO_ROOT / "test_merge_graph.json"
    with open(test_data_json, "r") as f:
        data = str(f.read())
    mock_manifest = {
        'createdOn': '05/02/2022 21:24:26',
        'runtime': 'python',
        'model': {
            'modelName': 'mergegraph0205',
            'serializedFile': '2022-01-29_22:56:55_296615_model_best.pickle',
            'handler': 'ggnn_classifier',
            'modelFile': 'ggnn_model.py',
            'modelVersion': '0.0.1'
        },
        'archiverVersion': '0.5.0'
    }
    ggnn.initialize(
        Context(
            model_name=None,
            model_dir=REPO_ROOT / "log/train-ggnn-model",
            manifest=mock_manifest,
            batch_size=1,
            gpu=1,
            mms_version='0.5.0',
            limit_max_image_pixels=True,
        )
    )
    start = time.time()
    pred = ggnn.inference(data)
    end = time.time()
    print("Prediction for test data: %s | Elapsed time: %f" %
          (pred, end - start))
