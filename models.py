import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from skorch import NeuralNetClassifier

class CNNClassifier(nn.Module):
    def __init__(self, config):
        super(CNNClassifier, self).__init__()

        self.transformer_device = config["transformer_device"]
        self.model_device = config["transformer_device"]
        self.transformer = config["bert"]

        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.transformer.embeddings.word_embeddings.embedding_dim,
                out_channels=config["nfilters"],
                kernel_size=kernel_size,
                stride=1) for kernel_size in config["kernels"]])

        self.pool_stride = config["pool_stride"]

        self.dropout = nn.Dropout(config["dropout"])
        self.fully_connected = nn.Linear(
            (config["nfilters"] // config["pool_stride"]) * len(config["kernels"]), 2)

        self.output_activation = (torch.sigmoid  # pylint: disable=no-member
                                  if config["output_activation"] == 'sigmoid'
                                  else F.softmax)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.transformer = self.transformer.to(
            torch.device(self.transformer_device))
        self.model_device = next(self.fully_connected.parameters()).device.type
        return self

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, x):
        m = (x > 0).int()
        x = self.transformer(x, attention_mask=m)[0]
        #
        x = x * m.unsqueeze(2)
        x = torch.where(x > 0, x, torch.tensor(-1.0))
        x = x.transpose(1, 2)
        seq_len = x.shape[-1]
        #
        if self.transformer_device != self.model_device:
            x = x.to(self.model_device)
        #
        x = [F.relu(conv(x)).transpose(1, 2) for conv in self.convolutions]
        x = [nn.functional.pad(i, (0, 0, 0, seq_len - i.shape[1])) for i in x]
        x = [F.max_pool1d(c, self.pool_stride) for c in x]
        x = torch.cat(x, dim=2)  # pylint: disable=no-member
        x = self.fully_connected(x)
        x = self.dropout(x)

        return self.output_activation(x)