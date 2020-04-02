# Shared task on semi-supervised identification of verbal MWE


## Libraries

* `Python 3.6 +`

* `numpy`
* `tensorflow`
* `keras`
* `scikit-learn`
* `pandas`

* Suggestion: install libraries in a `virtual environment`.

## Experiments

To run an experiment:

```
export MODEL_DIR=${HOME}/train_rnn_model/

mkdir -p $MODEL_DIR

python3 -u train_rnn_model.py \
  --train_dir=$MODEL_DIR \
  --model_name='model.ckpt' \
  --log_tensorboard=False \
  --max_epochs=100 \
  --early_stop_patience=10 \
  --early_stop_delta=0.001 \
  --embed_dim=100 \
  --spatial_dropout=0.4 \
  --n_layers=1 \
  --bilstm=False \
  --lstm_size=100 \
  --lstm_dropout=0.2 \
  --lstm_recurrent_dropout=0.2 \
  --batch_size=32 \
  --optimizer='adam' \
  --learning_rate=0.0001\
  --clipnorm=0.1
```

## TODOs:

* `Grid-search`