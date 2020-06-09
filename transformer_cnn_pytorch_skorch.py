#pylint: disable=invalid-name

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import BucketIterator

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report


from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

from models import CNNClassifier
from preprocess import load_tokenized_data, SentenceDataset, SkorchBucketIterator
from utils import build_model_name, convert_flags_to_dict, define_cnn_flags

from transformers import AutoModel


####

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

####
bert_type = 'distilbert-base-multilingual-cased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # pylint: disable=no-member
METRIC = "F1"

#####
(x_train, y_train), (x_val, y_val), (x_dev, y_dev) = load_tokenized_data(
    datafile='{}/data/{}.tokenized.pkl'.format(os.getcwd(), bert_type),
    language_codes=['DE', 'GA', 'HI', 'PT', 'ZH'],
    seed=SEED)

test_iterator = SkorchBucketIterator(
    dataset=SentenceDataset(data=(x_dev, y_dev)),
    batch_size=1,
    sort_key=lambda x: len(x.sentence),
    shuffle=False,
    device='cpu')

#####
transformer = AutoModel.from_pretrained(bert_type)
#####

config = {
    'nfilters': 128,
    'kernels': [1, 2, 3, 4, 5],
    'pool_stride': 3,
    'dropout': 0.2,
    'output_activation': 'sigmoid',
    'transformer_device': DEVICE,
    'bert': transformer
}

model = CNNClassifier(config)
model.to(DEVICE)   # pylint: disable=no-member
model.freeze_transformer()

from skorch import NeuralNetClassifier
from skorch.callbacks import Freezer
from skorch.callbacks import ProgressBar, EpochScoring, EarlyStopping, Checkpoint
from skorch.helper import predefined_split

net = NeuralNetClassifier(
    module=model,
    #
    iterator_train=SkorchBucketIterator,
    iterator_train__batch_size=32,
    iterator_train__sort_key=lambda x: len(x.sentence),
    iterator_train__shuffle=True,
    iterator_train__device=DEVICE,
    #
    iterator_valid=SkorchBucketIterator,
    iterator_valid__batch_size=32,
    iterator_valid__sort_key=lambda x: len(x.sentence),
    iterator_valid__shuffle=True,
    iterator_valid__device=DEVICE,

    train_split=predefined_split(SentenceDataset(data=(x_val, y_val))),

    optimizer=torch.optim.Adam,
    criterion=nn.BCELoss,
    callbacks=[
        ProgressBar(batches_per_epoch=len(x_train) // 128 + 1),
        EpochScoring(METRIC, lower_is_better=False),
        EarlyStopping(monitor=METRIC, patience=5),
        Checkpoint(monitor=METRIC)

    ],

    device=DEVICE
)

net.fit(SentenceDataset(data=(x_train, y_train)), y=None)