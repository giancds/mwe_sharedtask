from eval_scripts.evaluate import Main

def evaluate_model(net, test_iterator, tokenizer, args):
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
        tokens = tokenizer.convert_ids_to_tokens(x.detach().cpu().numpy().reshape(-1))
        # tokens = tokens
        y_pred = y_pred.cpu().detach().reshape(-1).tolist()
        for t, p in zip(tokens, y_pred):
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
    with open(args.dev_file, 'r') as dev:
        with open(args.dev_file.replace('dev.cupt', 'system.cupt'), 'w') as test:
            for line in dev:
                feats = line.split()
                if not line.startswith('#') and line is not '\n' and '-' not in feats[0]:
                    test.write(line.replace('*', str(preds[output_count])))
                    output_count += 1
                else:
                    test.write(line)

    _run_sript(args)

def _run_sript(args):

    args.debug = False
    args.combinatorial = True
    args.gold_file = open(args.dev_file, 'r')
    args.prediction_file = open(args.dev_file.replace('dev.cupt', 'system.cupt'), 'r')
    args.train_file = open(args.dev_file.replace('dev.cupt', 'train.cupt'), 'r')

    Main(args).run()