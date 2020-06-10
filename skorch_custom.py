import torch
import skorch
from sklearn.metrics import f1_score, classification_report
from torchtext.data import Dataset, Field, Example, BucketIterator
from tensorflow.keras.utils import to_categorical

class SkorchBucketIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            # We make a small modification: Instead of just returning batch
            # we return batch.text and batch.label, corresponding to X and y
            # if self.train:
            y =  batch.labels.to('cpu')
            y = to_categorical(y, num_classes=2)
            y = torch.tensor(y).to(self.device)
            batch.labels = y
            # else:
            #     batch.labels = batch.labels.float()
            yield batch.sentence, batch.labels


class SentenceDataset(Dataset):

    def __init__(self, data, min_len=5, **kwargs):
        self.min_len = min_len
        text_field = Field(use_vocab=False, pad_token=0, batch_first=True)
        label_field = Field(use_vocab=False, pad_token=0, batch_first=True)
        fields = [("sentence", text_field), ("labels", label_field)]
        examples = []
        for (x, y) in zip(data[0], data[1]):
            if len(x) < self.min_len:  # pad all sequences shorter than this
                x += [0] * (5 - len(x))
                y += [0] * (5 - len(y))
            examples.append(Example.fromlist([x, y], fields))
        super().__init__(examples, fields, **kwargs)


class IdiomClassifier(skorch.NeuralNetClassifier):

    def __init__(self, print_report=True, *args, **kwargs):
        self.print_report = print_report
        super(IdiomClassifier, self).__init__(*args, **kwargs)
        self.set_params(callbacks__valid_acc=None)


    def predict(self, X):
        self.module.eval()
        return torch.argmax(self.module(X), dim=2)


    def score(self, X, y=None):
        self.module.eval()
        ds = self.get_dataset(X)
        target_iterator = self.get_iterator(ds, training=False)

        y_true = []
        y_pred = []
        for x, y in target_iterator:
            preds = self.predict(x)
            y_pred.append(preds.view(-1))
            y = torch.argmax(y, dim=2)
            y_true.append(y.view(-1))
        y_true = torch.cat(y_true).detach().numpy()
        y_pred = torch.cat(y_pred).detach().numpy()

        if self.print_report:
            print(classification_report(y_true, y_pred))
        return f1_score(y_true, y_pred, average='binary')


class CustomScorer(skorch.callbacks.EpochScoring):
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        current_score = net.score(dataset_valid)
        self._record_score(net.history, current_score)
