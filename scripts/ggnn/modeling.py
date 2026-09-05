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
"""Modules that make up the pytorch GNN models."""
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


SMALL_NUMBER = 1e-8


def print_state_dict(mod):
    for n, t in mod.state_dict().items():
        print(n, t.size())


def num_parameters(mod) -> int:
    """Compute the number of trainable parameters in a nn.Module and its children.
    OBS:
        This function misses some parameters, i.e. in pytorch's official MultiheadAttention layer,
        while the state dict doesn't miss any!
    """
    num_params = sum(
        param.numel() for param in mod.parameters(recurse=True) if param.requires_grad
    )
    return f"{num_params:,} params, weights size: {num_params * 4 / 1e6:.3f}MB."


def assert_no_nan(tensor_list):
    for i, t in enumerate(tensor_list):
        assert not torch.isnan(t).any(), f"{i}: {tensor_list}"


################################################
# Main Model classes
################################################
class BaseGNNModel(nn.Module):
    def __init__(self):
        super().__init__()

    def setup(self, config, test_only, class_weights, gpu_id_to_use=None):
        if gpu_id_to_use is not None:
            self.dev = (
                torch.device("cuda:%s" % gpu_id_to_use) if torch.cuda.is_available(
                ) else torch.device("cpu")
            )
        else:
            self.dev = (
                torch.device("cuda") if torch.cuda.is_available(
                ) else torch.device("cpu")
            )
        print("Using GPU: %s" % self.dev)
        if class_weights is not None:
            self.loss = Loss(config, torch.Tensor(class_weights).to(self.dev))
        else:
            self.loss = Loss(config)

        # move model to device before making optimizer!
        self.to(self.dev)

        if test_only:
            self.opt = None
            self.eval()
        else:
            self.opt = self.get_optimizer(self.config)

    def get_optimizer(self, config):
        return optim.AdamW(self.parameters(), lr=config.lr)

    def num_parameters(self) -> int:
        """Compute the number of trainable parameters in this nn.Module and its children."""
        num_params = sum(
            param.numel()
            for param in self.parameters(recurse=True)
            if param.requires_grad
        )
        return f"{num_params:,} params, weights size: ~{num_params * 4 // 1e6:,}MB."

    def forward(
        self,
        vocab_ids,
        labels,
        edge_lists,
        selector_ids=None,
        pos_lists=None,
        num_graphs=None,
        graph_nodes_list=None,
        node_types=None,
        aux_in=None,
        test_time_steps=None,
        readout_mask=None,
        runtimes=None,
    ):
        # Input
        # selector_ids are ignored anyway by the NodeEmbeddings module that doesn't support them.
        raw_in = self.node_embeddings(vocab_ids, selector_ids)

        # GNN
        raw_out, raw_in, *unroll_stats = self.gnn(
            edge_lists, raw_in, pos_lists, node_types
        )  # OBS! self.gnn might change raw_in inplace, so use the two outputs
        # instead!

        # Readout
        if getattr(self.config, "has_graph_labels", False):
            assert (
                graph_nodes_list is not None and num_graphs is not None
            ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."
            nodewise_readout, graphwise_readout = self.readout(
                raw_in,
                raw_out,
                graph_nodes_list=graph_nodes_list,
                num_graphs=num_graphs,
                auxiliary_features=aux_in,
                readout_mask=readout_mask,
            )
            logits = graphwise_readout
        else:  # nodewise only
            nodewise_readout, _ = self.readout(
                raw_in, raw_out, readout_mask=readout_mask
            )
            graphwise_readout = None
            logits = nodewise_readout

        # do the old style aux_readout if not aux_use_better is set
        if getattr(self.config, "has_aux_input", False) and not getattr(
            self.config, "aux_use_better", False
        ):
            assert (
                self.config.has_graph_labels is True
            ), "Implementation hasn't been checked for use with aux_input and nodewise prediction! It could work or fail silently."
            assert aux_in is not None
            logits, graphwise_readout = self.aux_readout(logits, aux_in)

        if readout_mask is not None:  # need to mask labels in the same fashion.
            assert readout_mask.dtype == torch.bool, "Readout mask should be boolean!"
            labels = labels[readout_mask]

        # Metrics
        # accuracy, correct?, targets, maybe runtimes: actual, optimal
        metrics_tuple = self.metrics(logits, labels, runtimes)

        outputs = (logits,) + metrics_tuple + \
            (graphwise_readout,) + tuple(unroll_stats)

        return outputs


################################################
# GNN Encoder: Message+Aggregate, Update
################################################

# GNN Encoder, i.e. everything between input and readout.
# Will rely on the different msg+aggr and update modules to build up a GNN.


class GGNNEncoder(nn.Module):
    def __init__(self, config, readout=None):
        super().__init__()
        self.backward_edges = config.backward_edges

        self.gnn_layers = config.gnn_layers
        self.message_weight_sharing = config.message_weight_sharing
        self.update_weight_sharing = config.update_weight_sharing
        message_layers = self.gnn_layers // self.message_weight_sharing
        update_layers = self.gnn_layers // self.update_weight_sharing
        assert (
            message_layers * self.message_weight_sharing == self.gnn_layers
        ), "layer number and reuse mismatch."
        assert (
            update_layers * self.update_weight_sharing == self.gnn_layers
        ), "layer number and reuse mismatch."
        # self.layer_timesteps = config.layer_timesteps

        self.position_embeddings = config.position_embeddings

        # optional eval time unrolling parameter
        self.test_layer_timesteps = getattr(config, "test_layer_timesteps", 0)
        self.unroll_strategy = getattr(config, "unroll_strategy", "none")
        self.max_timesteps = getattr(config, "max_timesteps", 1000)
        self.label_conv_threshold = getattr(
            config, "label_conv_threshold", 0.995)
        self.label_conv_stable_steps = getattr(
            config, "label_conv_stable_steps", 1)

        # make readout avalable for label_convergence tests
        if self.unroll_strategy == "label_convergence":
            assert (
                not self.config.has_aux_input
            ), "aux_input is not supported with label_convergence"
            assert (
                readout
            ), "Gotta pass instantiated readout module for label_convergence tests!"
            self.readout = readout

        # Message and update layers
        self.message = nn.ModuleList()
        # for i in range(len(self.layer_timesteps)):§
        for i in range(message_layers):
            self.message.append(GGNNMessageLayer(config))

        self.update = nn.ModuleList()
        # for i in range(len(self.layer_timesteps)):
        for i in range(update_layers):
            self.update.append(GGNNUpdateLayer(config))

    def forward(
        self,
        edge_lists,
        node_states,
        pos_lists=None,
        node_types=None,
        test_time_steps=None,
    ):
        old_node_states = node_states.clone()

        if self.backward_edges:
            back_edge_lists = [x.flip([1]) for x in edge_lists]
            edge_lists.extend(back_edge_lists)

            # For backward edges we keep the positions of the forward edge!
            if self.position_embeddings:
                pos_lists.extend(pos_lists)

        # we allow for some fancy unrolling strategies.
        # Currently only at eval time, but there is really no good reason for this.
        assert (
            self.unroll_strategy == "none"
        ), "New layer_timesteps not implemented for this unroll_strategy."

        for i in range(self.gnn_layers):
            m_idx = i // self.message_weight_sharing
            u_idx = i // self.update_weight_sharing
            messages = self.message[m_idx](edge_lists, node_states, pos_lists)
            node_states = self.update[u_idx](messages, node_states, node_types)
        return node_states, old_node_states

    def label_convergence_forward(
        self, edge_lists, node_states, pos_lists, node_types, initial_node_states
    ):
        assert (
            len(self.layer_timesteps) == 1
        ), f"Label convergence only supports one-layer GGNNs, but {len(self.layer_timesteps)} are configured in layer_timesteps: {self.layer_timesteps}"

        stable_steps, i = 0, 0
        old_tentative_labels = self.tentative_labels(
            initial_node_states, node_states)

        while True:
            messages = self.message[0](edge_lists, node_states, pos_lists)
            node_states = self.update[0](messages, node_states, node_types)
            new_tentative_labels = self.tentative_labels(
                initial_node_states, node_states
            )
            i += 1

            # return the new node states if their predictions match the old node states' predictions.
            # It doesn't matter during testing since the predictions are the same anyway.
            stability = (
                (new_tentative_labels == old_tentative_labels)
                .to(dtype=torch.get_default_dtype())
                .mean()
            )
            if stability >= self.label_conv_threshold:
                stable_steps += 1

            if stable_steps >= self.label_conv_stable_steps:
                return node_states, i, True

            if i >= self.max_timesteps:  # maybe escape
                return node_states, i, False

            old_tentative_labels = new_tentative_labels

        raise ValueError("Serious Design Error: Unreachable code!")

    def tentative_labels(self, initial_node_states, node_states):
        logits, _ = self.readout(initial_node_states, node_states)
        preds = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(preds, dim=1)
        return predicted_labels


# Message Layers


class GGNNMessageLayer(nn.Module):
    """Implements the MLP message function of the GGNN architecture,
    optionally with position information embedded on edges.
    Args:
        edge_lists      (for each edge type) <M_i, 2>
        node_states     <N, D+S>
        pos_lists       <M> (optionally)
    Returns:
        incoming messages per node of shape <N, D+S>"""

    def __init__(self, config):
        super().__init__()
        self.edge_type_count = (
            config.edge_type_count * 2
            if config.backward_edges
            else config.edge_type_count
        )
        self.msg_mean_aggregation = config.msg_mean_aggregation
        self.dim = config.hidden_size

        self.transform = LinearNet(
            self.dim,
            self.dim * self.edge_type_count,
            bias=config.use_edge_bias,
            dropout=config.edge_weight_dropout,
        )

        self.pos_transform = None
        if getattr(config, "position_embeddings", False):
            self.selector_size = getattr(config, "selector_size", 0)
            self.emb_size = config.emb_size
            self.position_embs = PositionEmbeddings()

            self.pos_transform = LinearNet(
                self.dim,
                self.dim,
                bias=config.use_edge_bias,
                dropout=config.edge_weight_dropout,
            )

    def forward(self, edge_lists, node_states, pos_lists=None):
        """edge_lists: [<M_i, 2>, ...]"""

        # all edge types are handled in one matrix, but we
        # let propagated_states[i] be equal to the case with only edge_type i
        # propagated_states = (
        #    self.transform(node_states)
        #    .transpose(0, 1)
        #    .view(self.edge_type_count, self.dim, -1)
        # )
        propagated_states = self.transform(node_states).chunk(
            self.edge_type_count, dim=1
        )

        messages_by_targets = torch.zeros_like(node_states)
        if self.msg_mean_aggregation:
            device = node_states.device
            bincount = torch.zeros(
                node_states.size()[0], dtype=torch.long, device=device
            )

        for i, edge_list in enumerate(edge_lists):
            edge_targets = edge_list[:, 1]
            edge_sources = edge_list[:, 0]

            messages_by_source = torch.index_select(
                propagated_states[i], dim=0, index=edge_sources
            )

            if self.pos_transform:
                # print(i, len(pos_lists))
                pos_list = pos_lists[i]
                # torch.index_select(pos_gating, dim=0, index=pos_list)
                pos_by_source = self.position_embs(
                    pos_list.to(dtype=torch.get_default_dtype()),
                    self.emb_size,
                    dpad=self.selector_size,
                )

                pos_gating_by_source = 2 * torch.sigmoid(
                    self.pos_transform(pos_by_source)
                )

                # messages_by_source.mul_(pos_by_source)
                messages_by_source = messages_by_source * pos_gating_by_source

            messages_by_targets.index_add_(0, edge_targets, messages_by_source)

            if self.msg_mean_aggregation:
                bins = edge_targets.bincount(minlength=node_states.size()[0])
                bincount += bins

        if self.msg_mean_aggregation:
            divisor = bincount.float()
            divisor[bincount == 0] = 1.0  # avoid div by zero for lonely nodes
            # messages_by_targets /= divisor.unsqueeze_(1) + SMALL_NUMBER
            messages_by_targets = (
                messages_by_targets / divisor.unsqueeze_(1) + SMALL_NUMBER
            )

        return messages_by_targets


class PositionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positions, demb, dpad: int = 0):
        """Transformer-like sinusoidal positional embeddings.
                Args:
                position: 1d long Tensor of positions,
                demb: int    size of embedding vector
            """
        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, demb, 2.0,
                      device=positions.device) / demb)
        )

        sinusoid_inp = torch.ger(positions, inv_freq)
        pos_emb = torch.cat(
            (torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

        if dpad > 0:
            in_length = positions.size()[0]
            pad = torch.zeros((in_length, dpad))
            pos_emb = torch.cat([pos_emb, pad], dim=1)
            assert torch.all(
                pos_emb[:, -1] == torch.zeros(in_length)
            ), f"test failed. pos_emb: \n{pos_emb}"

        return pos_emb


# Update Layers


class GGNNUpdateLayer(nn.Module):
    """GRU update function of GGNN architecture, optionally distinguishing two kinds of node types.
    Args:
        incoming messages <N, D+S> (from message layer),
        node_states <N, D+S>,
        node_types <N> (optional)
    Returns:
        updated node_states <N, D+S>
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = config.graph_state_dropout
        # TODO(github.com/ChrisCummins/ProGraML/issues/27): Maybe decouple hidden
        # GRU size: make hidden GRU size larger and EdgeTrafo size non-square
        # instead? Or implement stacking gru layers between message passing steps.

        self.gru = nn.GRUCell(
            input_size=config.hidden_size, hidden_size=config.hidden_size
        )

        # currently only admits node types 0 and 1 for statements and identifiers.
        self.use_node_types = getattr(config, "use_node_types", False)
        if self.use_node_types:
            self.id_gru = nn.GRUCell(
                input_size=config.hidden_size, hidden_size=config.hidden_size
            )

    def forward(self, messages, node_states, node_types=None):
        if self.use_node_types:
            assert (
                node_types is not None
            ), "Need to provide node_types <N> if config.use_node_types!"
            output = torch.zeros_like(node_states, device=node_states.device)
            stmt_mask = node_types == 0
            output[stmt_mask] = self.gru(
                messages[stmt_mask], node_states[stmt_mask])
            id_mask = node_types == 1
            output[id_mask] = self.id_gru(
                messages[id_mask], node_states[id_mask])
        else:
            output = self.gru(messages, node_states)

        if self.dropout > 0.0:
            F.dropout(output, p=self.dropout,
                      training=self.training, inplace=True)
        return output


########################################
# GNN Output Layers
########################################


class Readout(nn.Module):
    """aka GatedRegression. See Eq. 4 in Gilmer et al. 2017 MPNN."""

    def __init__(self, config):
        super().__init__()
        self.has_graph_labels = config.has_graph_labels
        self.num_classes = config.num_classes
        self.use_tanh_readout = getattr(config, "use_tanh_readout", False)

        self.regression_gate = LinearNet(
            2 * config.hidden_size, self.num_classes, dropout=config.output_dropout,
        )
        self.regression_transform = LinearNet(
            config.hidden_size, self.num_classes, dropout=config.output_dropout,
        )

    def forward(
        self,
        raw_node_in,
        raw_node_out,
        graph_nodes_list=None,
        num_graphs=None,
        auxiliary_features=None,
        readout_mask=None,
    ):
        if readout_mask is not None:
            # mask first to only process the stuff that goes into the loss function!
            raw_node_in = raw_node_in[readout_mask]
            raw_node_out = raw_node_out[readout_mask]
            if graph_nodes_list is not None:
                graph_nodes_list = graph_nodes_list[readout_mask]

        gate_input = torch.cat((raw_node_in, raw_node_out), dim=-1)
        gating = torch.sigmoid(self.regression_gate(gate_input))
        if not self.use_tanh_readout:
            nodewise_readout = gating * self.regression_transform(raw_node_out)
        else:
            nodewise_readout = gating * torch.tanh(
                self.regression_transform(raw_node_out)
            )

        graph_readout = None
        if self.has_graph_labels:
            assert (
                graph_nodes_list is not None and num_graphs is not None
            ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."
            # aggregate via sums over graphs
            device = raw_node_out.device
            graph_readout = torch.zeros(
                num_graphs, self.num_classes, device=device)
            graph_readout.index_add_(
                dim=0, index=graph_nodes_list, source=nodewise_readout
            )
            if self.use_tanh_readout:
                graph_readout = torch.tanh(graph_readout)
        return nodewise_readout, graph_readout


class LinearNet(nn.Module):
    """Single Linear layer with WeightDropout, ReLU and Xavier Uniform
    initialization. Applies a linear transformation to the incoming data:
    :math:`y = xA^T + b`

    Args:
    in_features: size of each input sample
    out_features: size of each output sample
    bias: If set to ``False``, the layer will not learn an additive bias.
    Default: ``True``

    Shape:
    - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
    additional dimensions and :math:`H_{in} = \text{in\_features}`
    - Output: :math:`(N, *, H_{out})` where all but the last dimension
    are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0, gain=1.0):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.gain = gain
        self.test = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.test, gain=self.gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.dropout > 0.0:
            w = F.dropout(self.test, p=self.dropout, training=self.training)
        else:
            w = self.test
        return F.linear(input, w, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, dropout={}".format(
            self.in_features, self.out_features, self.bias is not None, self.dropout,
        )


###########################################
# Mixing in graph-level features to readout
###########################################


class AuxiliaryReadout(nn.Module):
    """Produces per-graph predictions by combining
    the per-graph predictions with auxiliary features.
    Note that this AuxiliaryReadout after Readout is probably a bad idea
    and BetterAuxiliaryReadout should be used instead."""

    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.aux_in_log1p = getattr(config, "aux_in_log1p", False)
        assert (
            config.has_graph_labels
        ), "We expect aux readout in combination with graph labels, not node labels"
        self.feed_forward = None

        self.batch_norm = nn.BatchNorm1d(
            config.num_classes + config.aux_in_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(
                config.num_classes + config.aux_in_size, config.aux_in_layer_size,
            ),
            nn.ReLU(),
            nn.Dropout(config.output_dropout),
            nn.Linear(config.aux_in_layer_size, config.num_classes),
        )

    def forward(self, graph_features, auxiliary_features):
        assert (
            graph_features.size()[0] == auxiliary_features.size()[0]
        ), "every graph needs aux_features. Dimension mismatch."
        if self.aux_in_log1p:
            auxiliary_features.log1p_()

        aggregate_features = torch.cat(
            (graph_features, auxiliary_features), dim=1)

        normed_features = self.batch_norm(aggregate_features)
        out = self.feed_forward(normed_features)
        return out, graph_features


class BetterAuxiliaryReadout(nn.Module):
    """Produces per-graph predictions by combining
    the raw GNN Encoder output with auxiliary features.
    The difference to AuxReadout(Readout()) is that the aux info
    is concat'ed before the nodewise readout and not after the
    reduction to graphwise predictions.
    """

    def __init__(self, config):
        super().__init__()

        self.aux_in_log1p = getattr(config, "aux_in_log1p", False)
        assert (
            config.has_graph_labels
        ), "We expect aux readout in combination with graph labels, not node labels"

        self.has_graph_labels = config.has_graph_labels
        self.num_classes = config.num_classes

        # now with aux_in concat'ed and batchnorm
        self.regression_gate = nn.Sequential(
            nn.BatchNorm1d(2 * config.hidden_size + config.aux_in_size),
            LinearNet(
                2 * config.hidden_size + config.aux_in_size,
                self.num_classes,
                dropout=config.output_dropout,
            ),
        )
        # now with aux_in concat'ed and with intermediate layer
        self.regression_transform = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size + config.aux_in_size),
            LinearNet(
                config.hidden_size + config.aux_in_size,
                config.aux_in_layer_size,
                dropout=config.output_dropout,
            ),
            nn.ReLU(),
            LinearNet(config.aux_in_layer_size, config.num_classes),
        )

    def forward(
        self,
        raw_node_in,
        raw_node_out,
        graph_nodes_list,
        num_graphs,
        auxiliary_features,
        readout_mask=None,
    ):
        assert (
            graph_nodes_list is not None and auxiliary_features is not None
        ), "need those"
        if readout_mask is not None:
            # mask first to only process the stuff that goes into the loss function!
            raw_node_in = raw_node_in[readout_mask]
            raw_node_out = raw_node_out[readout_mask]
            if graph_nodes_list is not None:
                graph_nodes_list = graph_nodes_list[readout_mask]

        if self.aux_in_log1p:
            auxiliary_features.log1p_()
        aux_by_node = torch.index_select(
            auxiliary_features, dim=0, index=graph_nodes_list
        )

        # info: the gate and regression include batch norm inside!
        gate_input = torch.cat(
            (raw_node_in, raw_node_out, aux_by_node), dim=-1)
        gating = torch.sigmoid(self.regression_gate(gate_input))
        trafo_input = torch.cat((raw_node_out, aux_by_node), dim=-1)
        nodewise_readout = gating * self.regression_transform(trafo_input)

        graph_readout = None
        if self.has_graph_labels:
            assert (
                graph_nodes_list is not None and num_graphs is not None
            ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."
            # aggregate via sums over graphs
            device = raw_node_out.device
            graph_readout = torch.zeros(
                num_graphs, self.num_classes, device=device)
            graph_readout.index_add_(
                dim=0, index=graph_nodes_list, source=nodewise_readout
            )
        return nodewise_readout, graph_readout


############################
# GNN Input: Embedding Layers
############################


class NodeEmbeddings(nn.Module):
    """Construct node embeddings from node ids
    Args:
    pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
    the Embedding. First dimension is being passed to Embedding as
    num_embeddings, second as embedding_dim.

    Forward
    Args:
    vocab_ids: <N, 1>
    Returns:
    node_states: <N, config.hidden_size>
    """

    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe LayerNorm and
    # Dropout on node_embeddings?
    # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Make selector embs
    # trainable?

    def __init__(self, config, pretrained_embeddings=None):
        super().__init__()
        self.inst2vec_embeddings = config.inst2vec_embeddings
        self.emb_size = config.emb_size

        if config.inst2vec_embeddings == "random":
            print("Initializing with random embeddings")
            self.node_embs = nn.Embedding(config.vocab_size, config.emb_size)
        else:
            raise NotImplementedError(config.inst2vec_embeddings)

    def forward(self, vocab_ids, *ignored_args, **ignored_kwargs):
        embs = self.node_embs(vocab_ids)

        return embs


class NodeEmbeddingsWithSelectors(NodeEmbeddings):
    """Construct node embeddings as content embeddings + selector embeddings.

    Args:
    pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
    the Embedding. First dimension is being passed to Embedding as
    num_embeddings, second as embedding_dim.

    Forward
    Args:
    vocab_ids: <N, 1>
    selector_ids: <N, 1>
    Returns:
    node_states: <N, config.hidden_size>
    """

    def __init__(self, config, pretrained_embeddings=None):
        super().__init__(config, pretrained_embeddings)

        self.node_embs = super().forward
        assert (
            config.use_selector_embeddings
        ), "This Module is for use with use_selector_embeddings!"

        selector_init = torch.tensor(
            # TODO(github.com/ChrisCummins/ProGraML/issues/27): x50 is maybe a
            # problem for unrolling (for selector_embs)?
            [[0, 50.0], [50.0, 0]],
            dtype=torch.get_default_dtype(),
        )
        self.selector_embs = nn.Embedding.from_pretrained(
            selector_init, freeze=True)

    def forward(self, vocab_ids, selector_ids):
        node_embs = self.node_embs(vocab_ids)
        selector_embs = self.selector_embs(selector_ids)
        embs = torch.cat((node_embs, selector_embs), dim=1)
        return embs


#############################
# Loss Accuracy Prediction
#############################


class Loss(nn.Module):
    """Cross Entropy loss with weighted intermediate loss, and
    L2 loss if num_classes is just 1.
    """

    def __init__(self, config, class_weights=None):
        super().__init__()
        self.config = config
        if config.num_classes == 1:
            self.loss = nn.MSELoss()
        else:
            # class labels '-1' don't contribute to the gradient!
            # however in most cases it will be more efficient to gather
            # the relevant data into a dense tensor
            self.loss = nn.CrossEntropyLoss(
                ignore_index=-1, reduction="mean", weight=class_weights)

    def forward(self, logits, targets):
        """inputs: (logits) or (logits, intermediate_logits)"""
        if self.config.num_classes == 1:
            l = torch.sigmoid(logits[0])
            logits = (l, logits[1])
        loss = self.loss(logits[0].squeeze(dim=1), targets)
        if getattr(self.config, "has_aux_input", False):
            loss = loss + self.config.intermediate_loss_weight * self.loss(
                logits[1], targets
            )
        return loss


class Metrics(nn.Module):
    """Common metrics and info for inspection of results.
    Args:
    logits, labels
    Returns:
    (accuracy, pred_targets, correct_preds, targets)"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, runtimes=None):
        # be flexible with 1hot labels vs indices
        if len(labels.size()) == 2:
            targets = labels.argmax(dim=1)
        elif len(labels.size()) == 1:
            targets = labels
        else:
            raise ValueError(
                f"labels={labels.size()} tensor is is neither 1 nor 2-dimensional. :/"
            )

        pred_targets = logits.argmax(dim=1)
        correct_preds = targets.eq(pred_targets).float()
        accuracy = torch.mean(correct_preds)

        ret = accuracy, correct_preds, targets

        if runtimes is not None:
            assert runtimes.size() == logits.size(), (
                f"We need to have a runtime for each sample and every possible label!"
                f"runtimes={runtimes.size()}, logits={logits.size()}."
            )
            # actual = runtimes[pred#torch.index_select(runtimes, dim=1, index=pred_targets)
            actual = torch.gather(
                runtimes, dim=1, index=pred_targets.view(-1, 1)
            ).squeeze()
            # actual = runtimes[:, pred_targets]
            optimal = torch.gather(
                runtimes, dim=1, index=targets.view(-1, 1)).squeeze()
            # optimal = runtimes[:, targets]
            ret += (actual, optimal)

        return ret


# Huggingface implementation
# perplexity = torch.exp(torch.tensor(eval_loss)), where loss is just the ave
