import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

def load_model(self, model_name, model_path):
        """Reload tensorflow session for saved model. Called by djinn.load,

        Args: 
            model_path (str, optional): Location of model if different than 
                       location set during training.
            model_name (str, optional): Name of model if different than 
                       name set during training.
        
        Returns: 
            Object: djinn regressor model.
            
        """
        # self.__sess = {}
        self.__models={}
        for p in range(0, self.__n_trees):
            tf.keras.backend.clear_session()
            # tf.reset_default_graph()
            # new_saver = \
            # tf.train.import_meta_graph('%s%s_tree%s.ckpt.meta'%(model_path,model_name,p))
            # self.__sess[p] = tf.Session()
            # new_saver.restore(self.__sess[p], '%s%s_tree%s.ckpt'%(model_path,model_name,p))
            self.__models[p] = tf.keras.models.load_model('%s%s_tree%s.ckpt'%(model_path,model_name,p),custom_objects={'WB_Init': WB_Init})
            # self.__models[p] = tf.keras.models.load_model('%s%s_tree%s.ckpt'%(model_path,model_name,p))
            print("Model %s restored"%p)

class WB_Init(tf.keras.initializers.Initializer):
    def __init__(self,dat=None,name=None):
        self.dat = dat
        self.name = name
        # print(type(dat))
        # print(dat)
                
    def __call__(self,shape,dtype):
        
        if not isinstance(self.dat,tf.Tensor):
            a = tf.convert_to_tensor(self.dat,dtype=tf.float32,name=self.name)
        else:
            # a = self.dat.value()
            a = self.dat
                    
        return a

try:
    from djinn_fns import tree_to_nn_weights, tf_dropout_regression, \
                    get_hyperparams, tf_continue_training, WB_Init
except:
    from djinn.djinn_fns import tree_to_nn_weights, tf_dropout_regression, \
                    get_hyperparams, tf_continue_training, WB_Init