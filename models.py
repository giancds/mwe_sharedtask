import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tensorflow.keras.utils import to_categorical


class CNNClassifier(nn.Module):
    def __init__(self, config):
        super(CNNClassifier, self).__init__()

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

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        x = self.transformer(x, attention_mask=mask)[0].transpose(1,2)
        seq_len = x.shape[-1]
        #
        x = [F.relu(conv(x)).transpose(1, 2) for conv in self.convolutions]
        x = [nn.functional.pad(i, (0, 0, 0, seq_len - i.shape[1])) for i in x]
        x = [F.max_pool1d(c, self.pool_stride) for c in x]
        x = torch.cat(x, dim=2)  # pylint: disable=no-member
        x = self.fully_connected(x)
        x = self.dropout(x)
        return self.output_activation(x)