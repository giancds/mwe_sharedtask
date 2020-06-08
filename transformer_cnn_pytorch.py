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

(x_train, y_train), (x_val, y_val), (x_dev, y_dev) = load_tokenized_data(
    datafile='{}/data/{}.tokenized.pkl'.format(os.getcwd(), bert_type),
    language_codes=['DE', 'GA', 'HI', 'PT', 'ZH'],
    seed=SEED)

train_iterator = SkorchBucketIterator(
    dataset=SentenceDataset(data=(x_train, y_train)),
    batch_size=32,
    sort_key=lambda x: len(x.sentence),
    shuffle=False,
    device=DEVICE)
valid_iterator = SkorchBucketIterator(
    dataset=SentenceDataset(data=(x_val, y_val)),
    batch_size=32,
    sort_key=lambda x: len(x.sentence),
    shuffle=False,
    device=DEVICE)

test_iterator = SkorchBucketIterator(
    dataset=SentenceDataset(data=(x_dev, y_dev)),
    batch_size=32,
    sort_key=lambda x: len(x.sentence),
    shuffle=False,
    device=DEVICE)

#####
transformer = AutoModel.from_pretrained(bert_type)
#####

config = {
    'nfilters': 128,
    'kernels': [1, 2, 3, 4, 5],
    'pool_stride': 3,
    'dropout': 0.2,
    'output_activation': 'sigmoid',
    'bert': transformer
}

model = CNNClassifier(config)
model.to(DEVICE)   # pylint: disable=no-member
model.freeze_transformer()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = nn.BCELoss()


def process_function(engine, batch):
    x, m, y = batch.sentence, batch.mask, batch.labels
    x = transformer(x, attention_mask=m)[0].transpose(1, 2)
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_function(engine, batch):
    x, m, y = batch.sentence, batch.mask, batch.labels
    x = transformer(x, attention_mask=m)[0].transpose(1, 2)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        return y_pred, y


trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
Loss(criterion).attach(train_evaluator, 'bce')

Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')
Loss(criterion).attach(validation_evaluator, 'bce')

pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])

def score_function(engine):
    val_loss = engine.state.metrics['bce']
    return -val_loss

handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iterator)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))

def log_validation_results(engine):
    validation_evaluator.run(valid_iterator)
    metrics = validation_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_bce = metrics['bce']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        .format(engine.state.epoch, avg_accuracy, avg_bce))
    pbar.n = pbar.last_print_n = 0

trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

checkpointer = ModelCheckpoint('/tmp/models', 'textcnn', n_saved=2, create_dir=True, save_as_state_dict=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'textcnn': model})

trainer.run(train_iterator, max_epochs=20)