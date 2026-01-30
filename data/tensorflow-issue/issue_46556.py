import tensorflow as tf

class MyRunConfig(tf.estimator.RunConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if '_distribute' in k:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

class MyDistributeStrategy(tf.distribute.MultiWorkerMirroredStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if '_extend' in k:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

strategy = MyDistributeStrategy()
config = MyRunConfig(train_distribute=strategy)

from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
CollectiveAllReduceExtended._enable_check_health = False