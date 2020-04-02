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


def extract_dataset(files):
    data = []
    for file in files:
        data += _build_dataset(open(file))
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