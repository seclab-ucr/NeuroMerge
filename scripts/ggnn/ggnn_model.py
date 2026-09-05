"""Seperate model module to comply with torch-model-archiver."""
"""https://github.com/pytorch/serve/blob/master/model-archiver/README.md"""

from modeling import (
    BaseGNNModel, 
    NodeEmbeddingsWithSelectors, 
    NodeEmbeddings, 
    BetterAuxiliaryReadout, 
    Readout, 
    AuxiliaryReadout, 
    GGNNEncoder, 
    Metrics,
)


class GGNNModel(BaseGNNModel):
    def __init__(self, config, pretrained_embeddings=None, test_only=False, class_weights=None, gpu_id_to_use=None):
        super().__init__()
        self.config = config

        # input layer
        if getattr(config, "use_selector_embeddings", False):
            self.node_embeddings = NodeEmbeddingsWithSelectors(
                config, pretrained_embeddings
            )
        else:
            self.node_embeddings = NodeEmbeddings(
                config, pretrained_embeddings)

        # Readout layer
        # get readout and maybe tack on the aux readout
        self.has_aux_input = getattr(self.config, "has_aux_input", False)
        self.aux_use_better = getattr(self.config, "aux_use_better", False)
        if self.has_aux_input and self.aux_use_better:
            self.readout = BetterAuxiliaryReadout(config)
        elif self.has_aux_input:
            self.readout = Readout(config)
            self.aux_readout = AuxiliaryReadout(config)
        else:
            assert not self.aux_use_better, "aux_use_better only with has_aux_input!"
            self.readout = Readout(config)

        # GNN
        # make readout available to label_convergence tests in GGNN Proper (at runtime)
        self.gnn = GGNNEncoder(config, readout=self.readout)

        # eval and training
        self.metrics = Metrics()

        self.setup(config, test_only, class_weights,
                   gpu_id_to_use=gpu_id_to_use)
        print(
            f"Number of trainable params in GGNNModel: {self.num_parameters()}")
