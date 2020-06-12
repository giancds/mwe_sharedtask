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


## Evaluation

To run the experiments and obtain the evaluation results at the end, you should copy the evaluation scripts from the oficial Shared-task repository [here](https://gitlab.com/parseme/sharedtask-data/tree/master/1.1/bin)and:
1. copy the files into a folder called `eval_scripts`
2. at the top of the `evaluate.py` replace `import tsvlib` with `from . import tsvlib`.

Otherwise, you should run with evaluation disabled

## Libraries

* `Python 3.6 +`

* `numpy`
* `scikit-learn`
* `pytorch`
* `huggingface transformers`
* `torchtext`
* `skorch`

* Suggestion: install libraries in a `virtual environment`.

## Experiments


### CNN Classifier

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/transformer/cnn

mkdir -p $MODEL_DIR

python3 -u transformer_cnn.py \
  --train_dir $MODEL_DIR \
  --bert_type 'distilbert-base-multilingual-cased' \
  --bert_device 'gpu' \
  --nfilters 128 \
  --kernels 1,2,3,4,5 \
  --pool_stride 5 \
  --dropout 0.2 \
  --output_activation 'sigmoid' \
  --batch_size 32 \
  --max_epochs 100 \
  --eval

```

### LSTM Classifier

To run an experiment (example with default hyperparameters):

```
export MODEL_DIR=${HOME}/transformer/rnn

mkdir -p $MODEL_DIR

python3 -u transformer_rnn.py \
  --train_dir $MODEL_DIR \
  --bert_type 'distilbert-base-multilingual-cased' \
  --bert_device 'gpu' \
  --nlayers 2 \
  --lstm_size 100 \
  --dropout 0.2 \
  --initrange 0.1 \
  --clipnorm 5.0 \
  --output_activation 'sigmoid' \
  --batch_size 32 \
  --max_epochs 100 \
  --eval
```

## TODOs:

* `Grid-search`
* NMT approach
