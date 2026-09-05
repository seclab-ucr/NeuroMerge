"""Data loader for GGNN."""
import csv
import enum
from pathlib import Path
from typing import Dict, Union
import os
import json
from collections import Counter
from sklearn.utils import shuffle
import sys

import torch
import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Batch

import programl as pg
from programl.proto.program_graph_pb2 import ProgramGraph

full_path = os.path.realpath(__file__)
REPO_ROOT = full_path.rsplit("NeuSE", maxsplit=1)[0] + "NeuSE"
ROOT_SCRIPT_DIR = REPO_ROOT + "/scripts"
sys.path.append(ROOT_SCRIPT_DIR)

from utils import parse_merge_json  # noqa

REPO_ROOT = Path(REPO_ROOT)

DATA_DIR = os.path.abspath(os.path.join(REPO_ROOT, "dataset"))
MERGE_RECORD_DIR = os.path.join(DATA_DIR, "merge-record")

PROGRAML_VOCABULARY = Path(os.path.abspath(
    os.path.join(REPO_ROOT, "dataset/vocab/programl.csv")))
MERGE_RECORD_DIR = Path(os.path.abspath(
    os.path.join(REPO_ROOT, "dataset/merge-record")))

MIN_MERGE_SIGNIFICANCE = 0.2


def load(file: Union[str, Path]) -> ProgramGraph:
    """Read a ProgramGraph protocol buffer from file.

    Args:
        file: The path of the ProgramGraph protocol buffer to load.
        cdfg: If true, convert the graph to CDFG during load.

    Returns:
        A ProgramGraph proto instance.
    """
    # Changed the way of loading (prior way seems to be buggy)
    graph = pg.load_graphs(Path(file))[0]

    return graph


def load_vocabulary(path: Path):
    """Read the vocabulary file used in the dataflow experiments."""
    vocab = {}
    with open(path) as f:
        vocab_file = csv.reader(f.readlines(), delimiter="\t")
        for i, row in enumerate(vocab_file, start=-1):
            if i == -1:  # Skip the header.
                continue
            (_, _, _, text) = row
            vocab[text] = i

    return vocab


class AblationVocab(enum.IntEnum):
    # No ablation - use the full vocabulary (default).
    NONE = 0
    # Ignore the vocabulary - every node has an x value of 0.
    NO_VOCAB = 1
    # Use a 3-element vocabulary based on the node type:
    #    0 - Instruction node
    #    1 - Variable node
    #    2 - Constant node
    NODE_TYPE_ONLY = 2


def filename(
    split: str, cdfg: bool = False, ablation_vocab: AblationVocab = AblationVocab.NONE
) -> str:
    """Generate the name for a data file.

    Args:
        split: The name of the split.
        cdfg: Whether using CDFG representation.
        ablate_vocab: The ablation vocab type.

    Returns:
        A file name which uniquely identifies this combination of
        split/cdfg/ablation.
    """
    name = str(split)
    if cdfg:
        name = f"{name}_cdfg"
    if ablation_vocab != AblationVocab.NONE:
        # transform if ablation_vocab was passed as int.
        if type(ablation_vocab) == int:
            ablation_vocab = AblationVocab(ablation_vocab)
        name = f"{name}_{ablation_vocab.name.lower()}"
    return f"{name}_data.pt"


def nx2data4serve(
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


def data2input4serve(data, config, dev):
    """Glue method that converts a batch from the dataloader into the input format of the model"""
    edge_lists = []
    edge_positions = (
        [] if getattr(config, "position_embeddings", False) else None
    )
    edge_indices = list(range(3))

    if config.ablate_structure:
        if config.ablate_structure == "control":
            edge_indices[0] = -1
        elif config.ablate_structure == "data":
            edge_indices[1] = -1
        elif config.ablate_structure == "call":
            edge_indices[2] = -1
        else:
            raise ValueError("unreachable")

    dev = (
        torch.device("cuda:%s" % dev) if torch.cuda.is_available(
        ) else torch.device("cpu")
    )

    for i in edge_indices:
        # mask by edge type
        mask = data.edge_attr[:, 0] == i  # <M_i>
        edge_list = data.edge_index[:, mask].t().to(dev)  # <et, M_i>
        edge_lists.append(edge_list)

        if getattr(config, "position_embeddings", False):
            edge_pos = data.edge_attr[mask, 1].to(dev)  # <M_i>
            edge_positions.append(edge_pos)

    data = Batch.from_data_list([data])

    inputs = {
        "vocab_ids": data.x[:, 0].to(dev),
        "edge_lists": edge_lists,
        "pos_lists": edge_positions,
        "node_types": data.x[:, 1].to(dev),
        "labels": torch.IntTensor([1]).to(dev),  # dummy
        "num_graphs": 1,  # only one sample per input
        "graph_nodes_list": data.batch.to(dev),
    }

    return inputs


def nx2data(
    graph: ProgramGraph,
    vocabulary: Dict[str, int],
    merge_record_json_dir: str,
    graph_fname: str,
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
    def get_graph_label(graph_fpath, merge_record_json_dir):
        merge_id = str(graph_fpath).split("/")[-1].split("-")[1]
        json_fpath = os.path.join(
            merge_record_json_dir, "data-" + merge_id + ".json")
        with open(json_fpath) as f:
            data = json.load(f)
        merged_time = data["Merged_exploration_time"]
        unmerged_time = data["Unmerged_exploration_time "]
        if merged_time >= unmerged_time:
            return 0
        else:
            return 1

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

    # Let us get the label from the JSON file
    merge_decision = get_graph_label(graph_fname, merge_record_json_dir)
    data_dict["y"] = merge_decision

    # branch prediction / profile info specific
    if not ignore_profile_info:
        raise NotImplementedError(
            "profile info is not supported with the new nx2data (from programgraph) adaptation."
        )

    # make Data
    data = Data(**data_dict)

    return data


class NeuSEDataset(InMemoryDataset):
    def __init__(
        self,
        root="dummy_path",
        split="dummy_split",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
        cdfg: bool = False,
        ablation_vocab: AblationVocab = AblationVocab.NONE,
        upsample_in_train: bool = False,
        filter_insignificant_merges: bool = True,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            cdfg: Use the CDFG graph format and vocabulary.
        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        self.cdfg = cdfg
        self.ablation_vocab = ablation_vocab
        self.upsample_in_train = upsample_in_train
        self.filter_insignificant_merges = filter_insignificant_merges
        super().__init__(root, transform, pre_transform)

        assert split in ["train", "val", "test"]
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        base = filename(self.split, self.cdfg, self.ablation_vocab)

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(
            self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 2

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / filename(
            self.split, self.cdfg, self.ablation_vocab
        )
        print("Full dataset path: %s" % full_dataset)
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # get vocab first
        vocab = load_vocabulary(PROGRAML_VOCABULARY)
        assert len(vocab) > 0, "vocab is empty :|"
        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")

        split_folder = ds_base / (self.split)
        assert split_folder.exists(), f"{split_folder} doesn't exist!"

        # collect .pb and call nx2data on the fly!
        print(
            f"=== DATASET {split_folder}: Collecting ProgramGraph.pb files into dataset"
        )

        # load files from filenames
        files = [x for x in split_folder.rglob("MergeGraph-*.pb")]

        if self.split == "train" and self.filter_insignificant_merges:
            print(
                "Filtering insignificant merges that do not exhibit significantly different exploration times...")
            filtered = []

            num_filered, total = 0, 0
            labels = []
            for fname in files:
                total += 1
                merge_id = str(fname).split("/")[-1].split("-")[1]
                merge_record_json_dir = MERGE_RECORD_DIR / \
                    Path("data-%s.json" % merge_id)
                merge = parse_merge_json(merge_record_json_dir)
                label = "MERGE" if merge.merged_time > merge.unmerged_time else "NOT_MERGE"
                if abs(merge.merged_time - merge.unmerged_time) / min(merge.merged_time, merge.unmerged_time) < MIN_MERGE_SIGNIFICANCE:
                    num_filered += 1
                    labels.append(label)
                    continue
                else:
                    filtered.append(fname)
                if total % 1000 == 0:
                    print(
                        f"Filtered {num_filered} insignificant merges out of {total} total merges | {Counter(labels)}.")

            files = filtered

        # upsample if set
        if self.split == "train" and self.upsample_in_train:
            upsampled = []

            # generate stats
            labels = []
            pos_files, neg_files = [], []
            for fname in files:
                label = str(fname).split("/")[-1].split("-")[-1].split(".")[0]
                if label == "1":
                    pos_files.append(fname)
                if label == "0":
                    neg_files.append(fname)
                labels.append(int(label))
            stats = Counter(labels)
            print(f"=== DATASET {split_folder}: Stats: {stats}")

            # assume that the 1 (SHOULD_MERGE) is the majority class
            diff = stats[1] - stats[0]

            upsampled = pos_files + neg_files
            for i in range(diff):
                upsampled.append(Path(neg_files[i % len(neg_files)]))

            shuffle(upsampled)

            # generate stats
            labels = []
            for fname in upsampled:
                label = str(fname).split("/")[-1].split("-")[-1].split(".")[0]
                labels.append(int(label))
            stats = Counter(labels)
            print(
                f"=== DATASET (AFTER UPSAMPLING) {split_folder}: Stats: {stats}")

            files = upsampled

        assert len(files) > 0, "no files collected. error."
        for file in tqdm.tqdm(files):
            if os.path.getsize(file) == 0:
                print(f"Skipping empty file: {file}")
                continue
            graph = load(file)
            data = nx2data(
                graph=graph,
                vocabulary=vocab,
                ablate_vocab=self.ablation_vocab,
                merge_record_json_dir=MERGE_RECORD_DIR,
                graph_fname=file,
            )
            data_list.append(data)

        print(
            f" * COMPLETED * === DATASET {split_folder}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {split_folder}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()
