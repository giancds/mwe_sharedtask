import torch
import torch.nn as nn
import torch.nn.functional as F

class NNClassifier(nn.Module):
    def __init__(self, config, transformer, transformer_device):
        super(NNClassifier, self).__init__()

        self.transformer_device = transformer_device
        self.model_device = transformer_device
        self.transformer = transformer
        self.dropout = nn.Dropout(config.dropout)
        self.ninputs = transformer.embeddings.word_embeddings.embedding_dim
        if config.hidden_size > 0:
            self.fully_connected1 = nn.Linear(self.ninputs, config.hidden_size)
            ninputs_to_classifier = config.hidden_size
        else:
            self.fully_connected1 = None
            ninputs_to_classifier = self.ninputs

        self.noutputs = 1
        if config.labels == 'multilabel':
            self.noutputs = config.num_outputs
        else:
            if config.output_activation == 'softmax':
                self.noutputs = 2

        self.fully_connected = nn.Linear(ninputs_to_classifier, self.noutputs)

        self.output_activation = ('sigmoid'  # pylint: disable=no-member
                                  if self.noutputs == 1
                                  else ('sigmoid'
                                        if config.output_activation == 'sigmoid'
                                        else 'softmax'))

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
        x = x.to(self.transformer_device)
        m = (x > 0).int()
        x = self.transformer(x, attention_mask=m)[0]
        #
        #
        if self.transformer_device != self.model_device:
            x = x.to(self.model_device)

        if self.fully_connected1 is not None:
            x = F.elu(self.fully_connected1(self.dropout(x)))

        return self.fully_connected(self.dropout(x))


class CNNClassifier(nn.Module):
    def __init__(self, config, transformer, transformer_device):
        super(CNNClassifier, self).__init__()

        self.transformer_device = transformer_device
        self.model_device = transformer_device
        self.transformer = transformer
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=transformer.embeddings.word_embeddings.embedding_dim,
                out_channels=config.nfilters,
                kernel_size=kernel_size,
                stride=1) for kernel_size in config.kernels])

        self.pool_stride = config.pool_stride
        self.dropout = nn.Dropout(config.dropout)

        ninputs = (config.nfilters // config.pool_stride) * len(config.kernels)

        self.noutputs = 1
        if config.labels == 'multilabel':
            self.noutputs = config.num_outputs
        else:
            if config.output_activation == 'softmax':
                self.noutputs = 2

        self.fully_connected = nn.Linear(ninputs, self.noutputs)

        self.output_activation = ('sigmoid'  # pylint: disable=no-member
                                  if self.noutputs == 1
                                  else ('sigmoid'
                                        if config.output_activation == 'sigmoid'
                                        else 'softmax'))

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
        x = x.to(self.transformer_device)
        m = (x > 0).int()
        x = self.transformer(x, attention_mask=m)[0].transpose(1, 2)
        #
        seq_len = x.shape[-1]
        #
        if self.transformer_device != self.model_device:
            x = x.to(self.model_device)
        #
        x = [F.elu(conv(x)).transpose(1, 2) for conv in self.convolutions]
        x = [nn.functional.pad(i, (0, 0, 0, seq_len - i.shape[1])) for i in x]
        x = [F.max_pool1d(c, self.pool_stride) for c in x]
        x = torch.cat(x, dim=2)  # pylint: disable=no-member

        return self.fully_connected(self.dropout(x))

class RNNClassifier(nn.Module):
    def __init__(self, config, transformer, transformer_device):
        super(RNNClassifier, self).__init__()

        self.transformer_device = transformer_device
        self.model_device = transformer_device
        self.transformer = transformer

        self.lstm = nn.LSTM(
            input_size=transformer.embeddings.word_embeddings.embedding_dim,
            hidden_size=config.lstm_size,
            num_layers=config.nlayers,
            batch_first=True,
            dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)
        self.noutputs = 1
        if config.labels == 'multilabel':
            self.noutputs = config.num_outputs
        else:
            if config.output_activation == 'softmax':
                self.noutputs = 2

        self.fully_connected = nn.Linear(config.lstm_size, self.noutputs)

        self.output_activation = ('sigmoid'  # pylint: disable=no-member
                                  if self.noutputs == 1
                                  else ('sigmoid'
                                        if config.output_activation == 'sigmoid'
                                        else 'softmax'))
        self.init_weights(config.initrange)

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
        x = x.to(self.transformer_device)
        m = (x > 0).int()
        x = self.transformer(x, attention_mask=m)[0]
        #
        if self.transformer_device != self.model_device:
            x = x.to(self.model_device)
        #
        x, _ = self.lstm(x)

        return self.fully_connected(self.dropout(x))

    def init_weights(self, initrange):
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                weight.data.uniform_(-initrange, initrange)

        self.fully_connected.bias.data.fill_(0)
        self.fully_connected.weight.data.uniform_(-initrange, initrange)


class CNNRNNClassifier(nn.Module):
    def __init__(self, config, transformer, transformer_device):
        super(CNNRNNClassifier, self).__init__()

        self.transformer_device = transformer_device
        self.model_device = transformer_device
        self.transformer = transformer

        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=transformer.embeddings.word_embeddings.embedding_dim,
                out_channels=config.nfilters,
                kernel_size=kernel_size,
                stride=1) for kernel_size in config.kernels])

        self.pool_stride = config.pool_stride

        ninputs = (config.nfilters // config.pool_stride) * len(config.kernels)
        self.lstm = nn.LSTM(
            input_size=ninputs,
            hidden_size=config.lstm_size,
            num_layers=config.nlayers,
            batch_first=True,
            dropout=config.dropout)

        self.dropout = nn.Dropout(config.dropout)

        self.noutputs = 1
        if config.labels == 'multilabel':
            self.noutputs = config.num_outputs
        else:
            if config.output_activation == 'softmax':
                self.noutputs = 2
        self.fully_connected = nn.Linear(config.lstm_size, self.noutputs)

        self.output_activation = ('sigmoid'  # pylint: disable=no-member
                                  if self.noutputs == 1
                                  else ('sigmoid'
                                        if config.output_activation == 'sigmoid'
                                        else 'softmax'))
        self.init_weights(config.initrange)

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
        x = x.to(self.transformer_device)
        m = (x > 0).int()
        x = self.transformer(x, attention_mask=m)[0].transpose(1, 2)
        #
        seq_len = x.shape[-1]
        if self.transformer_device != self.model_device:
            x = x.to(self.model_device)
        #
        x = [F.elu(conv(x)).transpose(1, 2) for conv in self.convolutions]
        x = [nn.functional.pad(i, (0, 0, 0, seq_len - i.shape[1])) for i in x]
        x = [F.max_pool1d(c, self.pool_stride) for c in x]
        x = torch.cat(x, dim=2)  # pylint: disable=no-member
        x = self.dropout(x)
        #
        x, _ = self.lstm(x)
        return self.fully_connected(self.dropout(x))

    def init_weights(self, initrange):
        for conv in self.convolutions:
            conv.weight.data.uniform_(-initrange, initrange)
            conv.bias.data.fill_(0)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                weight.data.uniform_(-initrange, initrange)
        self.fully_connected.weight.data.uniform_(-initrange, initrange)
        self.fully_connected.bias.data.fill_(0)