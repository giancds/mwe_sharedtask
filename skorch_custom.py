import torch
import skorch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchtext.data import Dataset, Field, Example, BucketIterator
from tensorflow.keras.utils import to_categorical


class SkorchBucketIterator(BucketIterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 sort_key=None,
                 device=None,
                 batch_size_fn=None,
                 train=True,
                 repeat=False,
                 shuffle=None,
                 sort=None,
                 sort_within_batch=None,
                 one_hot=True,
                 num_classes=2):
        self.one_hot = one_hot
        self.num_classes = num_classes
        super(SkorchBucketIterator,
              self).__init__(dataset, batch_size, sort_key, device,
                             batch_size_fn, train, repeat, shuffle, sort,
                             sort_within_batch)

    def __iter__(self):
        for batch in super().__iter__():
            # We make a small modification: Instead of just returning batch
            # we return batch.text and batch.label, corresponding to X and y
            # if self.train:
            if self.one_hot:
                y = batch.labels.to('cpu')
                y = to_categorical(y, num_classes=self.num_classes)
                y = torch.tensor(y).to(self.device)
                batch.labels = y
            else:
                batch.labels = batch.labels.float()
            yield batch.sentence, batch.labels


class SentenceDataset(Dataset):

    def __init__(self, data, min_len=5, **kwargs):
        self.min_len = min_len
        text_field = Field(use_vocab=False, pad_token=0, batch_first=True)
        label_field = Field(use_vocab=False, pad_token=-1, batch_first=True)
        fields = [("sentence", text_field), ("labels", label_field)]
        examples = []
        for (x, y) in zip(data[0], data[1]):
            if len(x) < self.min_len:     # pad all sequences shorter than this
                x += [0] * (5 - len(x))
                y += [-1] * (5 - len(y))
            examples.append(Example.fromlist([x, y], fields))
        super().__init__(examples, fields, **kwargs)


class IdiomClassifier(skorch.NeuralNetClassifier):

    def __init__(self, print_report=True, class_weights=None, score_average='binary', *args, **kwargs):
        self.print_report = print_report
        self.class_weights = class_weights
        self.score_average = score_average
        if class_weights is None:
            self.class_weights = [1.0, 1.0]
        super(IdiomClassifier, self).__init__(*args, **kwargs)
        self.set_params(callbacks__valid_acc=None)
        self.set_params(criterion__reduction='none')

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        if isinstance(self.criterion_, torch.nn.BCELoss):
            loss = super().get_loss(
                y_pred.view(-1), y_true.view(-1), X, *args, **kwargs)
        else:
            if isinstance(self.criterion_, torch.nn.NLLLoss):
                y_pred = self.module.output_activation(y_pred, dim=2)
            loss = super().get_loss(
                y_pred.view(-1, self.module.noutputs),
                y_true.long().view(-1), X, *args, **kwargs)

        if self.class_weights is not None:
            weights = torch.ones_like(y_true) * y_true
            for w, weight in enumerate(self.class_weights):
                weights = torch.where(
                    y_true == w,
                    torch.tensor(weight).float().to(self.device),
                    weights)
            loss = (loss * weights.view(-1))
        if isinstance(self.criterion_, torch.nn.BCELoss):
            mask = (y_true >= 0).int()
            loss = (loss * mask.view(-1))
        return loss.mean()

    def predict_proba(self, X):
        self.module.eval()
        y_pred = self.module(X)

        if self.module.output_activation == 'softmax':
            y_pred = F.softmax(y_pred, dim=2)
        else:
            y_pred = torch.sigmoid(y_pred)

        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)

        if self.module.noutputs > 1:
            y_pred = torch.argmax(y_pred, dim=2)
        else:
            y_pred = (y_pred > 0.5).int()

        return y_pred

    def score(self, X, y=None):
        self.module.eval()
        ds = self.get_dataset(X)
        target_iterator = self.get_iterator(ds, training=False)

        y_true = []
        y_pred = []
        for x, y in target_iterator:
            preds = self.predict(x)
            y_pred.append(preds.view(-1))
            if len(y.shape) > 2:
                y = torch.argmax(y, dim=2)
            y_true.append(y.view(-1))
        y_true = torch.cat(y_true).cpu().view(-1).detach().numpy().tolist()
        y_pred = torch.cat(y_pred).cpu().view(-1).detach().numpy().tolist()

        tt, tp = [], []
        for t, p in zip(y_true, y_pred):
            if t >= 0:
                tt.append(t)
                tp.append(p)

        y_true = tt
        y_pred = tp

        if self.print_report:
            print('Confusion matrix')
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred))
        return f1_score(y_true, y_pred, average=self.score_average)


class CustomScorer(skorch.callbacks.EpochScoring):

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        current_score = net.score(dataset_valid)
        self._record_score(net.history, current_score)