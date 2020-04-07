import pandas as pd


def process_cup(text):

    id_sent = None
    features = []

    for line in text:

        if line is '\n':
            id_sent = None
        elif line.startswith('# source_sent_id'):
            tokens = line.split()
            id_sent = tokens[-1]
        elif not line.startswith('#'):
            feats = line.split()

            feats_dict = {
                'id_sent': id_sent,
                'id': feats[0],
                'form': feats[1],
                'lemma': feats[2],
                'upos': feats[3],
                'xpos': feats[4],
                'feats': feats[5],
                'head': feats[6],
                'deprel': feats[7],
                'deps': feats[8],
                'misc': feats[9],
                'mwe': feats[10]
            }
            features.append(feats_dict)

    return pd.DataFrame(features)


def extract_dataset(files, per_word=False):
    data = []
    processing_func = _build_per_word_dataset if per_word else _build_dataset
    for file in files:
        data += processing_func(open(file))
    return data


def _build_dataset(text):
    examples = []
    flag = False
    example = ''
    for line in text:
        if line is '\n':     # if it is an empty line, we reset everything
            label = 1 if flag else 0
            examples.append((example.strip(), label))
            example = ''
            flag = False

        elif not line.startswith('#'):     # if it is not a line of metadata
            feats = line.split()
            example += ' ' + feats[3]
            if feats[10] is not '*' and flag == False:
                flag = True

    return examples


def _build_per_word_dataset(text):
    examples = []
    example = ''
    labels = []
    for line in text:
        if line is '\n':     # if it is an empty line, we reset everything
            examples.append((example.strip(), labels))
            example = ''
            labels = []

        elif not line.startswith('#'):     # if it is not a line of metadata
            feats = line.split()
            example += ' ' + feats[3]
            label = 0 if feats[10] is '*' else 1
            labels.append(label)

    return examples


def build_model_name(FLAGS):
    name = ('sentlevel_{0}epochs.{1}-{2}eStop.{3}embDim.{4}-{5}dropout.{6}-{7}-{8}lstm.'
            '{9}lstmDrop.{10}lstmRecDrop.{11}-{12}.'
            '{14}Loss.{15}batch.{16}.{17}lr.{18}-{19}decay.{20}norm.'
            '{21}initScale.ckpt').format(
                FLAGS.max_epochs,
                FLAGS.early_stop_patience,
                FLAGS.early_stop_delta,
                FLAGS.embed_dim,
                FLAGS.dropout,
                'spatial-' if FLAGS.spatial_dropout else '',
                FLAGS.n_layers,
                FLAGS.lstm_size,
                'bi-' if FLAGS.bilstm else '',
                FLAGS.lstm_dropout,
                FLAGS.lstm_recurrent_dropout,
                FLAGS.output_size,
                FLAGS.output_activation,
                (FLAGS.output_threshold + 'outThresh.') if FLAGS.output_size == 1 and  FLAGS.output_activation == 'sigmoid' else '',
                FLAGS.loss_function,
                FLAGS.batch_size,
                FLAGS.optimizer,
                FLAGS.learning_rate,
                FLAGS.lr_decay,
                FLAGS.start_decay,
                FLAGS.clipnorm,
                FLAGS.init_scale
                )
    return name
