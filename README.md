# Shared task on semi-supervised identification of verbal MWE

## Data

To run the experiments we need to create a folder called `data` and copy the data folders containing the datasets for the selected languages.

You can obtain the data from the Shared-task page [here](https://gitlab.com/parseme/sharedtask-data/tree/master/1.2).

So far we're using:

* Chinese (ZH)
* German (DE)
* Hindi (HI)
* Irish (GA)
* Portuguese (PT)


## Libraries

* `Python 3.6 +`

* `numpy`
* `tensorflow`
* `keras`
* `scikit-learn`
* `pandas`

* Suggestion: install libraries in a `virtual environment`.

## Experiments


### Sentence Level

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_rnn_model/

mkdir -p $MODEL_DIR

python3 -u train_rnn_model.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --feature='upos' \
  --embed_dim=100 \
  --dropout=0.1 \
  --spatial_dropout=False \
  --n_layers=1 \
  --bilstm=False \
  --lstm_size=100 \
  --lstm_dropout=0.2 \
  --lstm_recurrent_dropout=0.0 \
  --output_activation='sigmoid' \
  --output_size=2 \
  --output_threshold=0.5 \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='sgd' \
  --learning_rate=1.0\
  --lr_decay=0.5 \
  --start_decay=6\
  --clipnorm=5.0\
  --init_scale=0.05\
  --verbose=2
```

**Note**: `output_threshold` is only relevant when using in combination with `--output_activation='sigmoid'` and `--output_size=1`. Otherwise it is irrelevant.

### Per word Level

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_perword_model/

mkdir -p $MODEL_DIR

python3 -u train_perword_model.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --feature='upos' \
  --embed_dim=100 \
  --dropout=0.1 \
  --spatial_dropout=False \
  --n_layers=1 \
  --bilstm=False \
  --lstm_size=100 \
  --lstm_dropout=0.2 \
  --lstm_recurrent_dropout=0.0 \
  --output_activation='sigmoid' \
  --output_size=2 \
  --output_threshold=0.5 \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='sgd' \
  --learning_rate=1.0\
  --lr_decay=0.8 \
  --start_decay=6\
  --clipnorm=5.0\
  --init_scale=0.05\
  --verbose=2
```

**Note**: `output_threshold` is only relevant when using in combination with `--output_activation='sigmoid'` and `--output_size=1`. Otherwise it is irrelevant.

## Ensembles

### Sentence Level

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_rnn_model/

mkdir -p $MODEL_DIR

python3 -u train_rnn_boosting.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --n_estimators=50 \
  --boost_lr=1.0 \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --feature='upos' \
  --embed_dim=100 \
  --dropout=0.1 \
  --spatial_dropout=False \
  --n_layers=1 \
  --bilstm=False \
  --lstm_size=100 \
  --lstm_dropout=0.2 \
  --lstm_recurrent_dropout=0.0 \
  --output_activation='sigmoid' \
  --output_size=2 \
  --output_threshold=0.5 \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='sgd' \
  --learning_rate=1.0\
  --lr_decay=0.5 \
  --start_decay=6\
  --clipnorm=5.0\
  --init_scale=0.05\
  --verbose=2
```

**Note**: `output_threshold` is only relevant when using in combination with `--output_activation='sigmoid'` and `--output_size=1`. Otherwise it is irrelevant.

### Sentence Level

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_rnn_model/

mkdir -p $MODEL_DIR

python3 -u train_perword_boosting.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --n_estimators=50 \
  --boost_lr=1.0 \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --feature='upos' \
  --embed_dim=50 \
  --dropout=0.1 \
  --spatial_dropout=False \
  --n_layers=1 \
  --bilstm=False \
  --lstm_size=100 \
  --lstm_dropout=0.2 \
  --lstm_recurrent_dropout=0.0 \
  --output_activation='sigmoid' \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='sgd' \
  --learning_rate=1.0\
  --lr_decay=0.5 \
  --start_decay=6\
  --clipnorm=5.0\
  --init_scale=0.05\
  --verbose=2
```

**Note**: `output_threshold` and `--output_size=1` are not available for this classifier. This is a limitation of using `AdaboostClassifier` available in `scikit-learn`for implement the boosting method.


## CNNs

### SentLevel

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_cnn_model/

mkdir -p $MODEL_DIR

python3 -u train_cnn_model.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --embed_dim=30 \
  --emb_dropout=0.1 \
  --spatial_dropout=True \
  --filters=128 \
  --ngram=3 \
  --global_pooling='max' \
  --n_layers=1 \
  --dense_size=50 \
  --dropout=0.1 \
  --output_activation='sigmoid' \
  --output_size=2 \
  --output_threshold=0.5 \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='adam' \
  --learning_rate=0.0001\
  --lr_decay=0.5 \
  --start_decay=0\
  --clipnorm=1.0 \
  --init_scale=0.1 \
  --verbose=1 \
  --feature='upos+deprel'
```

### SentLevel Boosting

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/train_cnn_model/

mkdir -p $MODEL_DIR

python3 -u train_cnn_boosting.py \
  --train_dir=$MODEL_DIR \
  --log_tensorboard=False \
  --n_estimators=50 \
  --boost_lr=1.0 \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --embed_dim=30 \
  --emb_dropout=0.1 \
  --spatial_dropout=True \
  --filters=128 \
  --ngram=3 \
  --global_pooling='max' \
  --n_layers=1 \
  --dense_size=50 \
  --dropout=0.1 \
  --output_activation='sigmoid' \
  --output_size=2 \
  --output_threshold=0.5 \
  --loss_function='binary_crossentropy' \
  --weighted_loss=True \
  --batch_size=32 \
  --optimizer='adam' \
  --learning_rate=0.0001\
  --lr_decay=0.5 \
  --start_decay=0\
  --clipnorm=1.0 \
  --init_scale=0.1 \
  --verbose=1 \
  --feature='upos+deprel'
```

## TODOs:

* `Grid-search`
* NMT approach for per word labeling
* Transformers
