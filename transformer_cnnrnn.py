import argparse
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar, EarlyStopping, Checkpoint
from skorch.helper import predefined_split
from torchtext.data import Dataset, Field, Example, BucketIterator

from tensorflow.keras.utils import to_categorical

from transformers import AutoModel, AutoTokenizer

from models import CNNRNNClassifier
from preprocess import load_tokenized_data
from skorch_custom import SkorchBucketIterator, CustomScorer, IdiomClassifier, SentenceDataset
from evaluation import evaluate_model
from utils import build_model_name

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     # pylint: disable=no-member
LANGUAGE_CODES = ['DE', 'GA', 'HI', 'PT', 'ZH']
CWD = os.getcwd()
BASE_DIR = os.path.expanduser("~")      # this will point to the user's home
TRAIN_DIR = "transformer/cnn-rnn"


parser = argparse.ArgumentParser(description='Classifier using CNNs')
parser.add_argument(
    '--bert_type',
    type=str,
    default='distilbert-base-multilingual-cased',
    help='transormer model [should be a miltilingual model]')
parser.add_argument(
    '--bert_device',
    type=str,
    default='gpu',
    help='device to run the transformer model')
parser.add_argument(
    '--metric',
    type=str,
    default='f1',
    help='sklearn metric to evaluate the model while training')
parser.add_argument(
    '--nfilters',
    type=int,
    default=128,
    help='number of convolution filters')
parser.add_argument(
    '--kernels',
    type=list,
    default=[1, 2, 3, 4, 5],
    help='number of convolution filters')
parser.add_argument(
    '--pool_stride',
    type=int,
    default=5,
    help='size of the stride for the pooling operation')
parser.add_argument(
    '--nlayers',
    type=int,
    default=1,
    help='number of convolution filters')
parser.add_argument(
    '--lstm_size',
    type=int,
    default=200,
    help='number of convolution filters')
parser.add_argument(
    '--dropout',
    type=float,
    default=0.5,
    help='dropout probability for the dense layer')
parser.add_argument(
    '--initrange',
    type=float,
    default=0.05,
    help='range to initialize the lstm layers')
parser.add_argument(
    '--clipnorm',
    type=float,
    default=1.0,
    help='limit to clip the l2 norm of gradients')
parser.add_argument(
    '--output_activation',
    type=str,
    default='sigmoid',
    help='output activation')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='training batch size')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='validation/evaluation batch size')
parser.add_argument(
    '--max_epochs',
    type=int,
    default=100,
    help='max number of epochs to train the model')
parser.add_argument(
    "--train_dir",
    type=str,
    default=os.path.join(BASE_DIR, TRAIN_DIR) + "/",
    help="Train dir")
parser.add_argument(
    "--eval",
    action="store_true",
    help="eval at the end of the training process")

args = parser.parse_args()
args.kernels = [int(i) for i in args.kernels if ',' not in str(i)]
transformer_device = torch.device(
    'cuda' if torch.cuda.is_available() and args.bert_device == 'gpu'
    else 'cpu')
# transformer_device

(x_train, y_train), (x_val, y_val), (x_dev, y_dev) = load_tokenized_data(
    datafile='data/{}.tokenized.pkl'.format(args.bert_type),
    language_codes=LANGUAGE_CODES,
    seed=SEED)

targets = np.concatenate(y_train).reshape(-1)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(targets),
                                     y=targets)

tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
transformer = AutoModel.from_pretrained(args.bert_type)

model = CNNRNNClassifier(args, transformer, transformer_device)
model_name = build_model_name(args, model='cnn-rnn')
model.to(DEVICE)     # pylint: disable=no-member
model.freeze_transformer()

progress_bar = ProgressBar(batches_per_epoch=len(x_train) // args.batch_size + 1)
scorer = CustomScorer(scoring=None, name="F1", lower_is_better=False, use_caching=False)
early_stopping =  EarlyStopping(monitor='F1', patience=20, lower_is_better=False)
checkpoint = Checkpoint(
    monitor='F1_best',
    dirname=args.train_dir,
    f_params='{}.params.pt'.format(model_name),
    f_optimizer='{}.optimizer.pt'.format(model_name),
    f_history='{}.history.json'.format(model_name))

net = IdiomClassifier(
    module=model,
    class_weights=class_weights,
    print_report=False,
     #
    iterator_train=SkorchBucketIterator,
    iterator_train__batch_size=args.batch_size,
    iterator_train__sort_key=lambda x: len(x.sentence),
    iterator_train__shuffle=True,
    iterator_train__device=DEVICE,
    iterator_train__one_hot=args.output_activation == 'softmax',
     #
    iterator_valid=SkorchBucketIterator,
    iterator_valid__batch_size=args.eval_batch_size,
    iterator_valid__sort_key=lambda x: len(x.sentence),
    iterator_valid__shuffle=True,
    iterator_valid__device=DEVICE,
    iterator_valid__one_hot=args.output_activation == 'softmax',

    train_split=predefined_split(SentenceDataset(data=(x_val, y_val))),
    optimizer=torch.optim.Adam,
    criterion=nn.BCELoss,
    callbacks=[progress_bar, scorer, early_stopping, checkpoint],
    device=DEVICE,
)

net.fit(SentenceDataset(data=(x_train, y_train)), y=None, epochs=args.max_epochs)

net.initialize()
net.load_params(checkpoint=checkpoint)

print(model_name)
args.eval = True
if args.eval:
    for code in LANGUAGE_CODES:
        print('#' * 20)
        print('# Evaluating Language: {}'.format(code))
        print('#' * 20)
        test_iterator = SkorchBucketIterator(
            dataset=SentenceDataset(data=(x_dev[code], y_dev[code])),
            batch_size=args.eval_batch_size,
            sort=False,
            sort_key=lambda x: len(x.sentence),
            shuffle=False,
            train=False,
            one_hot=args.output_activation == 'softmax',
            device=DEVICE)
        args.dev_file = 'data/{}/dev.cupt'.format(code)
        evaluate_model(net, test_iterator, tokenizer, args)

print("#" * 20)
print("\nTraining finished!!!")