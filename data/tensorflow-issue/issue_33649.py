from tensorflow.keras.callbacks import Callback

class Metrics(Callback):
    def __init__(self, dev_data, classifier, dataloader):
        self.best_f1_score = 0.0
        self.dev_data = dev_data
        self.classifier = classifier
        self.predictor = Predictor(classifier, dataloader)
        self.dataloader = dataloader

    def on_epoch_end(self, epoch, logs=None):
        print("start to evaluate....")
        _, preds = self.predictor(self.dev_data)
        y_trues, y_preds = [self.dataloader.label_vector(v["label"]) for v in self.dev_data], preds
        f1 = f1_score(y_trues, y_preds, average="weighted")
        print(classification_report(y_trues, y_preds,
                                    target_names=self.dataloader.vocab.labels))
        if f1 > self.best_f1_score:
            self.best_f1_score = f1
            self.classifier.save_model()
            print("best metrics, save model...")