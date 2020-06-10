import sys
import argparse
from scripts.evaluate import Main

def evaluate_model(net, test_iterator, tokenizer, dev_file):
    preds = []
    sents = []
    i = 0
    for x, y in test_iterator:
        y_pred = net.predict(x)
    #     i += 1
        if i % 40 == 0:
            print(i)
        i += 1
        sub_tokens = []
        sub_preds = []
        text = []
        predictions = []
        tokens = tokenizer.convert_ids_to_tokens(x.detach().numpy().reshape(-1))
        for t, p in zip(tokens, y_pred.detach().reshape(-1).tolist()):
            if '#' in t:
                sub_tokens.append(t.replace('#', ''))
                sub_preds.append(p)
            else:
                if sub_tokens:
                    old_token = ''.join([text[-1]] + sub_tokens)
                    old_pred = sum(sub_preds)
                    text = text[0:-1]
                    text.append(old_token)
                    predictions = predictions[0:-1]
                    predictions.append(old_pred if old_pred == 0 else 1)
                    old_token = t
                    old_pred = p
                    sub_tokens = []
                    sub_preds = []
                else:
                    old_token = t
                    old_pred = p
                text.append(old_token)
                predictions.append(old_pred)
                assert len(text[1:-1]) == len(predictions[1:-1])
        sents.append(text[1:-1])
        preds += predictions[1:-1]

    output_count = 0
    with open(dev_file, 'r') as dev:
        with open(dev_file.replace('dev.cupt', 'system.cupt'), 'w') as test:
            for line in dev:
                if not line.startswith('#') and line is not '\n':
                    test.write(line.replace('*', str(preds[output_count])))
                    output_count += 1
                else:
                    test.write(line)

    _run_sript(dev_file)

def _run_sript(dev_file):

    parser = argparse.ArgumentParser(description="""
            Evaluate input `prediction` against `gold`.""")
    parser.add_argument("--debug", action="store_true",
            help="""Print extra debugging information (you can grep it,
            and should probably pipe it into `less -SR`)""")
    parser.add_argument("--combinatorial", action="store_true",
            help="""Run O(n!) algorithm for weighted bipartite matching.
            You should probably not use this option""")
    parser.add_argument("--train", metavar="train_file", dest="train_file",
            required=False, type=argparse.FileType('r'),
            help="""The training file in .cupt format (to calculate
            statistics regarding seen MWEs)""")
    parser.add_argument("--gold", metavar="gold_file", dest="gold_file",
            required=True, type=argparse.FileType('r'),
            help="""The reference gold-standard file in .cupt format""")
    parser.add_argument("--pred", metavar="prediction_file",
            dest="prediction_file", required=True,
            type=argparse.FileType('r'),
            help="""The system prediction file in .cupt format""")

    sys.argv.extend(['--gold', dev_file,
                    '--pred', dev_file.replace('dev.cupt', 'system.cupt'),
                    '--train', dev_file.replace('dev.cupt', 'train.cupt')])
    Main(parser.parse_args()).run()