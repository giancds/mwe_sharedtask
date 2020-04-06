def lr_scheduler(epoch, lr, lr_decay, start_decay):
    lr_decay = config.lr_decay **  max(epoch - config.start_decay, 0.0)
    return lt * lr_decay