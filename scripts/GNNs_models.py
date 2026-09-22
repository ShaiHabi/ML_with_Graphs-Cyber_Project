# Shai Habi, based on HW2, 26/08/2026.
# This file contains the implementations of the GNN models.


# General libraries
import numpy as np

# PyTorch:
import torch
import torch.nn as nn
import torch.nn.functional as F

# Python Geometric
from torch_geometric.nn import (GCNConv, GINConv, GATConv, TransformerConv,
                                global_mean_pool)

# Sklearn evaluation metrics:
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

class GNN_Model(torch.nn.Module):

  def __init__(self, GNN_type, dropout, input_dim, hidden_dim=64, output_dim=1, num_layers=5, heads=None):
      super().__init__()

      # Save general configuration
      self.GNN_type = GNN_type.upper()
      self.dropout = dropout
      self.hidden_dim = hidden_dim
      self.output_dim = output_dim
      self.num_layers = num_layers
      self.heads = heads

      # Validates model type
      supported_models = {"GCN", "GIN", "GAT", "GT"}
      if self.GNN_type not in supported_models:
          raise ValueError(f"Unsupported GNN type: {GNN_type}. Check supported_models")

      # Validates number of layers
      if self.num_layers < 1:
          raise ValueError("num_layers must be at least 1.")

      # Only attention-based models use heads
      if self.GNN_type in {"GAT", "GT"} and (self.heads is None or self.heads < 1):
          raise ValueError(f"{self.GNN_type} requires the 'heads' parameter to be a positive integer.")

      # GNN layers and BatchNorm layers between GNN layers
      self.convs = torch.nn.ModuleList()
      self.bns = torch.nn.ModuleList()

      # If there is only one GNN layer:
      # input_dim to hidden_dim
      if num_layers == 1:
          self.convs.append(self.build_conv(input_dim, hidden_dim))

      else:
          # First layer: input_dim to hidden_dim
          self.convs.append(self.build_conv(input_dim, hidden_dim))

          # Other layers: hidden_dim to hidden_dim
          # The last layer is hidden_dim as well, since we do graph classification
          # and we need the node embeddings for pooling
          for i in range(num_layers - 1):
              self.convs.append(self.build_conv(hidden_dim, hidden_dim))

          # BatchNorm after every layer except the last
          for i in range(num_layers - 1):
              self.bns.append(nn.BatchNorm1d(hidden_dim))


      # Graph-level classification:
      # Converts all node embeddings of each graph
      # into one graph embedding using mean aggregation
      self.pool = global_mean_pool

      # Converts the graph embedding to the final output
      self.linear = nn.Linear(hidden_dim, output_dim)





  def build_conv(self, input_dim, output_dim):
      """
      Builds and returns the appropriate GNN convolution layer based on the selected GNN type.

      Inputs:
      --- input_dim: int
      --- output_dim: int
      Output:
      --- conv: torch.nn.Module
      """

      # GCN
      if self.GNN_type == "GCN":
          return GCNConv(input_dim, output_dim)

      # GIN
      elif self.GNN_type == "GIN":
          # Each GINConv uses an MLP
          mlp = nn.Sequential(
              nn.Linear(input_dim, output_dim),
              nn.ReLU(),
              nn.Linear(output_dim, output_dim)
          )
          return GINConv(
              nn=mlp,
              eps=0.0, # As the MalNet paper does
              train_eps=False
          )

      # GAT
      elif self.GNN_type == "GAT":
          return GATConv(
              input_dim,
              output_dim,
              heads=self.heads,
              concat=False
          )

      # GT - the graph transformer of Shi et al. 2021, "Masked Label Prediction".
      # Despite the name it is a message passing layer like GAT, not a global
      # attention layer: attention is computed over each node's incoming edges
      # only, so the cost stays linear in |E| rather than in |V| squared.
      # What differs from GAT is the form of the attention - a scaled dot product
      # between separate query and key projections, instead of GAT's additive score
      # over the concatenated pair.
      elif self.GNN_type == "GT":
          return TransformerConv(
              input_dim,
              output_dim,
              heads=self.heads,
              concat=False       # average the heads, exactly as GAT does above
          )                      # dropout omitted for the same reason as in GAT above


  def forward(self, batched_data):
    """
    Performs a forward pass from node features to graph-level predictions.
    Assumption:
    --- loss_func is BCEWithLogitsLoss which gets as input logits (so no need to finish with sigmoid)
    Input:
    --- batched_data: PyG Batch object
    Output:
    --- out: torch.Tensor
    """

    # Extract graph data from the PyG batch
    x = batched_data.x.float()
    edge_index = batched_data.edge_index
    batch = batched_data.batch

    # All layers except the last:
    # Conv -> BatchNorm -> ReLU -> Dropout
    for i in range(self.num_layers - 1):
        x = self.convs[i](x, edge_index)
        x = self.bns[i](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

    # Last GNN layer returns the final node embeddings
    x = self.convs[self.num_layers - 1](x, edge_index)

    # Converts all node embeddings of each graph
    # into one graph embedding
    graph_embedding = self.pool(x, batch)

    # Converts the graph embedding to the final prediction
    out = self.linear(graph_embedding)

    return out


# Training implementation
# The loss function should be created outside this function.
# For imbalanced training sets, pos_weight must be calculated only from the current training set.

def train_one_epoch_GNN_model(model, device, data_loader, optimizer, loss_fn):
    """
    Trains the given GNN model for one epoch.

    Inputs:
    --- model: torch.nn.Module
    --- device: torch.device
    --- data_loader: torch_geometric.loader.DataLoader
    --- optimizer: torch.optim.Optimizer
    --- loss_fn: torch.nn.Module
    Output:
    --- average_loss: float
    """

    model.train()
    total_loss = 0
    total_graphs = 0

    for batch in data_loader:
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
          continue

        else:
          # Mask as sanity-check for graphs without a valid label:
          # ignore nan targets (unlabeled) when computing training loss
          is_labeled = batch.y == batch.y

          # Runs the model:
          optimizer.zero_grad()
          out = model(batch)

          # Filters predictions and labels using the labeled mask
          predictions = out[is_labeled]
          labels = batch.y[is_labeled].type(torch.float32)
          labels = labels.view_as(predictions) # Fits labels shape to the model output

          # Calculates the batch's loss
          loss = loss_fn(predictions, labels)
          loss.backward()
          optimizer.step()

          # Number of graphs that actually participated in the loss
          num_labeled = is_labeled.sum().item()

          # Updates the total avg. loss
          total_loss += loss.item() * num_labeled
          total_graphs += num_labeled


    if total_graphs == 0:
        raise ValueError("The training DataLoader contained no usable labeled graphs.")
    average_loss = total_loss / total_graphs
    return average_loss


def evaluate_GNN_model(model, device, data_loader, loss_fn):
    """
    Evaluates the GNN model on a given dataset.
    For both validating and testing - implementation of the evaluation function

    Inputs:
    --- model: torch.nn.Module
    --- device: torch.device
    --- data_loader: torch_geometric.loader.DataLoader
    --- loss_fn: torch.nn.Module
    Output:
    --- results: dict of floats
    """

    model.eval()

    # Sets the settings:
    y_true = []
    y_pred = []
    y_prob = []
    total_loss = 0
    total_graphs = 0

    # Runs batches through the model:
    for batch in data_loader:
        batch = batch.to(device)

        # No single-node guard here, unlike the training loop above. That guard is
        # there because BatchNorm cannot form a batch statistic out of one value and
        # raises in train mode. Under model.eval() BatchNorm uses its running
        # statistics instead, so a batch holding a single one-node graph evaluates
        # fine. Skipping such a batch would drop that graph from every metric while
        # num_test_graphs in the results CSV still counted it - a silent disagreement
        # between a score and the set it claims to score.
        with torch.no_grad():

            # Mask as sanity-check for graphs without a valid label:
            # ignore nan targets (unlabeled) when computing eval loss/metrics
            is_labeled = batch.y == batch.y

            # Runs the model
            out = model(batch)

            # Filters predictions and labels using the labeled mask
            out = out[is_labeled]
            labels = batch.y[is_labeled].view(out.shape).type(torch.float32)

            # Calculates the loss without updating the model
            loss = loss_fn(out, labels)

            # Converts logits to probabilities
            probabilities = torch.sigmoid(out)

            # Converts probabilities to binary predictions
            predictions = (probabilities >= 0.5).long()

        # Accumulates the loss over all graphs
        num_labeled = is_labeled.sum().item()
        total_loss += loss.item() * num_labeled
        total_graphs += num_labeled

        # Saves true labels, predictions and probabilities
        y_true.append(labels.detach().cpu())
        y_pred.append(predictions.detach().cpu())
        y_prob.append(probabilities.detach().cpu())

    # Out of the for loop's scope
    if total_graphs == 0:
        raise ValueError("The evaluation DataLoader contained no usable labeled graphs.")

    # Combines the results from all batches
    y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)
    y_prob = torch.cat(y_prob, dim=0).numpy().reshape(-1)

    # Calculates average loss
    average_loss = total_loss / total_graphs

    # Calculates evaluation metrics
    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_prob)

    results = {
          "loss": average_loss,
          "accuracy": accuracy_score(y_true, y_pred),
          "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
          "binary_f1": f1_score(y_true, y_pred, zero_division=0),
          "precision": precision_score(y_true, y_pred, zero_division=0),
          "recall": recall_score(y_true, y_pred, zero_division=0),
          "roc_auc": auc
    }
    return results
