# pylint: disable=invalid-name, missing-module-docstring
import pickle
import numpy as  np

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from preprocess import load_word_dataset


tf.compat.v1.flags.DEFINE_string(
    "model_type",
    # 'bert-base-multilingual-cased',
    'distilbert-base-multilingual-cased',
    "Model type to extract the embeddings")

tf.compat.v1.flags.DEFINE_string(
    "pooling_type",
    'bert',
    "Type of averaging to apply to the embbedings")

FLAGS = tf.compat.v1.flags.FLAGS

print('Loading models')

tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_type)
model = TFAutoModel.from_pretrained(FLAGS.model_type)

data = {}


# TODO: return to one-by-one
def embedd(language_code):

    train_sentences, train_labels = load_word_dataset(
        ['data/{}/train.cupt'.format(language_code)], train=True)

    dev_sentences, dev_labels = load_word_dataset(
        ['data/{}/dev.cupt'.format(language_code)], train=False)

    print('Train {} - Max.Len. {}'.format(
        len(train_sentences), max([len(l.split()) for l in train_sentences])))
    print('Dev {} - Max.Len. {}'.format(
        len(dev_sentences), max([len(l.split()) for l in dev_sentences])))


    _x_train = []
    for i, sentence in enumerate(train_sentences):
        if i % 100 == 0:
            print('Step {}'.format(i+1))
        input_ids = tokenizer.encode(
            sentence,
            add_special_tokens=True,
            return_tensors='tf')
        output = model(input_ids)
        if FLAGS.pooling_type == 'average':
            output = tf.keras.layers.GlobalAveragePooling1D()(output[0]).numpy()
        else:
            output = output[0][0].numpy()[0,:].reshape(1, -1)
        _x_train.append(output)
    _x_train = np.concatenate(_x_train, axis=0)
    _y_train = np.array(train_labels)

    _x_dev = []
    for i, sentence in enumerate(dev_sentences):
        if i % 100 == 0:
            print('Step {}'.format(i+1))
        input_ids = tokenizer.encode(
            sentence,
            add_special_tokens=True,
            return_tensors='tf')
        output = model(input_ids)
        output = tf.keras.layers.GlobalAveragePooling1D()(output[0]).numpy()
        _x_dev.append(output)
    _x_dev = np.concatenate(_x_dev, axis=0)
    _y_dev = np.array(dev_labels)

    return _x_train, _y_train, _x_dev, _y_dev



#####
# German (DE)
#####

code = 'DE'
print('German')
x_train, y_train, x_dev, y_dev = embedd(code)
with open('{}.{}.embdata.pkl'.format(code, FLAGS.model_type), 'wb') as f:
    pickle.dump({
        'x_train': x_train,
        'y_train': y_train,
        'x_dev': x_dev,
        'y_dev': y_dev
    }, f)

del x_train
del y_train
del x_dev
del y_dev


# #####
# # Irish (GA)
# #####
print('Irish')

code = 'GA'
x_train, y_train, x_dev, y_dev = embedd(code)
with open('{}.{}.embdata.pkl'.format(code, FLAGS.model_type), 'wb') as f:
    pickle.dump({
        'x_train': x_train,
        'y_train': y_train,
        'x_dev': x_dev,
        'y_dev': y_dev
    }, f)

del x_train
del y_train
del x_dev
del y_dev


# # #####
# # # Hindi (HI)
# # #####
print('Hindi')
code = 'HI'
x_train, y_train, x_dev, y_dev = embedd(code)
with open('{}.{}.embdata.pkl'.format(code, FLAGS.model_type), 'wb') as f:
    pickle.dump({
        'x_train': x_train,
        'y_train': y_train,
        'x_dev': x_dev,
        'y_dev': y_dev
    }, f)

del x_train
del y_train
del x_dev
del y_dev


# # #####
# # # Portuguese (PT)
# # #####

print('Portuguese')

code = 'PT'
x_train, y_train, x_dev, y_dev = embedd(code)
with open('{}.{}.embdata.pkl'.format(code, FLAGS.model_type), 'wb') as f:
    pickle.dump({
        'x_train': x_train,
        'y_train': y_train,
        'x_dev': x_dev,
        'y_dev': y_dev
    }, f)

del x_train
del y_train
del x_dev
del y_dev


# # #####
# # # Chinese (ZH)
# # #####

print('Chinese')
code = 'ZH'
x_train, y_train, x_dev, y_dev = embedd(code)
with open('{}.{}.embdata.pkl'.format(code, FLAGS.model_type), 'wb') as f:
    pickle.dump({
        'x_train': x_train,
        'y_train': y_train,
        'x_dev': x_dev,
        'y_dev': y_dev
    }, f)

del x_train
del y_train
del x_dev
del y_dev


