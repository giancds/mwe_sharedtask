""" Utilities module. """


def build_model_name(args, model='rnn-cnn'):
    name = ''
    if model == 'rnn':
        name = ("{0}.{1}.{2}layers.{3}lstm.{4}dropout.{5}init.{6}activation"
                "{7}clipnorm.{8}batch.{9}epochs".format(
                    args.bert_type, args.metric, args.nlayers, args.lstm_size,
                    args.dropout, args.initrange, args.output_activation,
                    args.clipnorm, args.batch_size, args.max_epochs
                ))
    elif model == 'cnn':
        name = ("{0}.{1}.{2}filters.{3}kernels.{4}poolstride.{5}dropout."
                "{6}activation.{7}batch.{8}epochs".format(
                    args.bert_type, args.metric, args.nfilters, args.kernels,
                    args.pool_stride, args.dropout, args.output_activation,
                    args.batch_size, args.max_epochs
                ))
    elif model == 'nn':
        name = ("{0}.{1}.{2}hidden.{3}dropout.{4}activation.{5}batch.{6}epochs".format(
                    args.bert_type, args.metric, args.hidden_size, args.dropout,
                    args.output_activation, args.batch_size, args.max_epochs
                ))
    elif model == 'cnn-rnn':
         name = ("{0}.{1}.{2}filters.{3}kernels.{4}poolstride.{5}layers."
                "{6}lstm.{7}dropout.{8}init.{9}activation.{10}batch."
                "{11}epochs".format(
                    args.bert_type, args.metric, args.nfilters, args.kernels,
                    args.pool_stride, args.nlayers, args.lstm_size, args.dropout,
                    args.initrange, args.output_activation, args.batch_size,
                    args.max_epochs
                ))
    return name