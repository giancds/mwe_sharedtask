""" Utilities module. """


def build_model_name(args, rnn=False):
    name = ''
    if rnn:
        name = ("{0}.{1}.{2}layers.{3}lstm.{4}dropout.{5}init.{6}activation"
                "{7}clipnorm.{8}batch.{9}epochs".format(
                    args.bert_type, args.metric, args.nlayers, args.lstm_size,
                    args.dropout, args.initrange, args.output_activation,
                    args.clipnorm, args.batch_size, args.max_epochs
                ))
    else:
        name = ("{0}.{1}.{2}filters.{3}kernels.{4}poolstride.{5}dropout."
                "{6}activation.{7}batch.{8}epochs".format(
                    args.bert_type, args.metric, args.nfilters, args.kernels,
                    args.pool_stride, args.dropout, args.output_activation,
                    args.batch_size, args.max_epochs
                ))
    return name