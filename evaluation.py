from eval_scripts.evaluate import Main

ID_LABELS = {
    0: '*',
    1: 'IAV',
    2: 'IRV',
    3: 'LVC.cause',
    4: 'LVC.full',
    5: 'LS.ICV',
    6: 'MVC',
    7: 'VID',
    8: 'VPC.full',
    9: 'VPC.semi',
}

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
                    predictions.append(old_pred
                                       if old_pred == 0
                                       else (sub_preds[0]
                                             if sub_preds[0] > 0
                                             else max(sub_preds)))
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

    binary = args.labels == 'binary'
    output_count = 0
    with open(args.dev_file, 'r') as dev:
        with open(args.dev_file.replace('dev.cupt', 'temp.cupt'), 'w') as temp:
            for line in dev:
                feats = line.split()
                if not line.startswith('#') and line != '\n' and '-' not in feats[0]:
                    prediction = preds[output_count]
                    if prediction == 0:
                        label = '*'
                    else:
                        label = ID_LABELS.get(prediction, '*')
                        # label = 1
                    new_line = '\t'.join(
                        [str(f) for f in feats[0:-1]] + [str(label)] + ['\n'])
                    temp.write(new_line)
                    output_count += 1
                else:
                    temp.write(line)

    # post-process the file to get the predictions into cupt format
    with open(args.dev_file.replace('dev.cupt', 'temp.cupt'), 'r') as temp:
        with open(args.dev_file.replace('dev.cupt', 'system.cupt'), 'w') as test:
            current_prediction = [1, None]
            verb_found = False
            for line in temp:
                feats = line.split('\t')
                if not line.startswith('#') and line != '\n' and '-' not in feats[0]:

                    if feats[10] == '*':
                        test.write(line)
                        # print(line)
                    else:

                        if current_prediction[1] is None:

                            label = '{}:{}'.format(current_prediction[0], feats[10])
                            # label = str(current_prediction[0])
                            verb_found = True if feats[3] == 'VERB' else False
                            current_prediction[1] = feats[10]

                        else:

                            if feats[10] == current_prediction[1]:

                                if verb_found and feats[3] != 'VERB':
                                    label = current_prediction[0]

                                elif verb_found and feats[3] == 'VERB':
                                    current_prediction[0] = current_prediction[0] + 1
                                    current_prediction[1] = feats[10]
                                    label = '{}:{}'.format(current_prediction[0], feats[10])
                                    # label = str(current_prediction[0])

                                elif not verb_found:
                                    label = current_prediction[0]
                                    verb_found = True if feats[3] == 'VERB' else False

                            else:
                                current_prediction[0] = current_prediction[0] + 1
                                current_prediction[1] = feats[10]
                                label = '{}:{}'.format(current_prediction[0], feats[10])
                                # label = str(current_prediction[0])
                                verb_found = True if feats[3] == 'VERB' else False
                        new_line = '\t'.join(feats[0:-2] + [str(label)] + ['\n'])
                        test.write(new_line)
                        # print(new_line)
                else:
                    if line == '\n':
                        current_prediction = [1, None]
                        verb_found = False
                    test.write(line)
                    # print(line)

    if args.eval:
        _run_sript(args)

def _run_sript(args):

    args.debug = False
    args.combinatorial = True
    args.gold_file = open(args.dev_file, 'r')
    args.prediction_file = open(args.dev_file.replace('dev.cupt', 'system.cupt'), 'r')
    args.train_file = open(args.dev_file.replace('dev.cupt', 'train.cupt'), 'r')
    args.debug = False
    print('\n\nRunning shared-task eval script\n\n')
    Main(args).run()