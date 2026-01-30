import numpy as np

warnings.warn('`model.predict_classes()` is deprecated and '
                  'will be removed after 2021-01-01. '
                  'Please use instead:'
                  '* `np.argmax(model.predict(x), axis=-1)`, '
                  '  if your model does multi-class classification '
                  '  (e.g. if it uses a `softmax` last-layer activation).'
                  '* `(model.predict(x) > 0.5).astype("int32")`, '
                  '  if your model does binary classification '
                  '  (e.g. if it uses a `sigmoid` last-layer activation).')