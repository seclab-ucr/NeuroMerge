# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configs"""
import dataset


class ProGraMLBaseConfig(object):
    def __init__(self):
        self.name = self.__class__.__name__

        # Training Hyperparameters
        self.num_epochs = 50
        self.batch_size = 64
        # limit the number of nodes per batch to a sensible maximum
        # by possibly discarding certain samples from the batch.
        self.max_num_nodes = 200000
        self.lr: float = 0.00005
        self.patience = 10000
        self.clip_grad_norm: float = 0.0
        self.train_subset = [0, 100]
        self.random_seed: int = 888

        # Readout
        self.output_dropout: float = 0.0

        # Model Hyperparameters
        self.emb_size: int = 128
        self.edge_type_count: int = 3

        self.vocab_size: int = 643
        self.cdfg_vocab: bool = False

        # ABLATION OPTIONS
        # NONE = 0 No ablation - use the full vocabulary (default).
        # NO_VOCAB = 1 Ignore the vocabulary - every node has an x value of 0.
        # NODE_TYPE_ONLY = 2 Use a 3-element vocabulary based on the node type:
        #    0 - Instruction node
        #    1 - Variable node
        #    2 - Constant node
        # 0 NONE, 1 NO_VOCAB, 2 NODE_TYPE_ONLY
        self.ablation_vocab: dataset.AblationVocab = 0

        # inst2vec_embeddings can now be 'none' as well!
        # this reduces the tokens that the network sees to only
        # !IDENTIFIERs and !UNK statements
        #  One of {zero, constant, random, random_const, finetune, none}
        self.inst2vec_embeddings = "random"

        self.ablate_structure = None  # one of {control,data,call}

    @classmethod
    def from_dict(cls, params):
        """instantiate Config from params dict that overrides default values where given."""
        config = cls()
        if params is None:
            return config

        for key in params:
            if hasattr(config, key):
                setattr(config, key, params[key])
            else:
                print(
                    f"(*CONFIG FROM DICT*  Default {config.name} doesn't have a key {key}. Will add key to config anyway!"
                )
                setattr(config, key, params[key])
        return config

    def to_dict(self):
        config_dict = {
            a: getattr(self, a)
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        }
        return config_dict

    def check_equal(self, other):
        # take either config object or config_dict
        other_dict = other if isinstance(other, dict) else other.to_dict()
        if not self.to_dict() == other_dict:
            print(
                f"WARNING: GGNNConfig.check_equal() FAILED:\nself and other are unequal: "
                f"The difference is {set(self.to_dict()) ^ set(other.to_dict())}.\n self={self.to_dict()}\n other={other_dict}"
            )


class GGNN_NeuSE_Config(ProGraMLBaseConfig):
    def __init__(self):
        super().__init__()
        ###############
        # Model Hyperparameters
        self.gnn_layers: int = 8
        self.message_weight_sharing: int = 2
        self.update_weight_sharing: int = 2

        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types = True
        self.use_edge_bias: bool = True
        self.position_embeddings: bool = True

        # Aggregate by mean or by sum
        self.msg_mean_aggregation: bool = True
        self.backward_edges: bool = True

        ###############
        # Regularization
        self.edge_weight_dropout: float = 0.0
        self.graph_state_dropout: float = 0.2

        ###############
        # Dataset inherent, don't change!
        self.num_classes: int = 2
        self.has_graph_labels: bool = True
        self.has_aux_input: bool = False

        # self.use_selector_embeddings: bool = False
        # self.selector_size: int = 2 if getattr(self, 'use_selector_embeddings', False) else 0
        # TODO(Zach) Maybe refactor non-rectangular edge passing matrices for independent hidden size.
        # hidden size of the whole model
        self.hidden_size: int = self.emb_size + \
            getattr(self, "selector_size", 0)
