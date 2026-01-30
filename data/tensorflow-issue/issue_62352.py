from tensorflow import keras

import tensorflow as tf                                                         # load tensorflow 
ckpt=tf.keras.callbacks.ModelCheckpoint('/content/')                            # make check point 
ckpt_params=ckpt.__dict__                                                       # get all parameters 
print('Parameters:',ckpt_params)                                                # see parameters dictionary
ckpt_params['save_best_only']=True                                              # update parameter "save_best_only" to True (default False)
ckpt_params['save_weights_only']=True                                           # update parameter "save_weights_only" to True (default False)
ckpt.set_params(ckpt_params)                                                    # try to make update parameters
print('Is same as updated?',ckpt_params==ckpt.__dict__)                         # make check both are same or not 
# both look same but if you look carefully `ckpt_params` and `ckpt.__dict__` will contain copy of themself in new key `'params'`
print('Parameters:',ckpt_params)                                                # see parameters dictionary

import tensorflow as tf                                                         # load tensorflow 
ckpt=tf.keras.callbacks.ModelCheckpoint('/content/')                            # make check point 
ckpt_params=ckpt.__dict__.copy()                                                # get all parameters 
print('Parameters:',ckpt_params)                                                # see parameters dictionary
ckpt_params['save_best_only']=True                                              # update parameter "save_best_only" to True (default False)
ckpt_params['save_weights_only']=True                                           # update parameter "save_weights_only" to True (default False)
#ckpt_params_deep_copy={key:value for key,value in ckpt_params.items()}          # make deep copy of `ckpt_params`
ckpt.set_params(ckpt_params)                                                    # try to make update parameters
print('Is same as updated?',ckpt_params==ckpt.__dict__)                         # make check both are same or not 
# both look same but if you look carefully `ckpt_params` and `ckpt.__dict__` will contain copy of themself in new key `'params'`
print('Uncomman parameters:',[key for key in ckpt.__dict__ if key not in ckpt_params])# see parameters dictionary
print('Uncomman parameters:',ckpt.__dict__['params'])                           # which is copy of itself only

import tensorflow as tf                                                         # load tensorflow 
lstp=tf.keras.callbacks.EarlyStopping()                                         # make check point 
lstp_params=lstp.__dict__.copy()                                                # get all parameters 
print('Parameters        :',lstp_params)                                        # see parameters dictionary
lstp_params['patience']=10                                                      # update parameter "patience" to 10 (default 0)
lstp_params['verbose']=1                                                        # update parameter "verbose" to 1 (default 0)
print('Updated parameters:',lstp_params)                                        # see updated parameters 
lstp.set_params(lstp_params.copy())                                             # try to make update parameters
print('Is same as updated?',lstp_params==lstp.__dict__)                         # make check both are same or not 
# Even values are not updated from `lstp_params` and `lstp.__dict__` will contain copy of themself in new key `'params'`
print('Object parameters :',lstp.__dict__)                                              # see parameters dictionary

import tensorflow as tf                                                         # load tensorflow 
import warnings                                                                 # load warnings 
class UpdatedModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

  # define set parameters method 
  def set_params(self,params:dict)->None:

    ''' make set give parameters dictionary ''' 

    if isinstance(params,dict):                                                 # make check parameter must be dictionary 
      for param,value in params.items():                                        # get parameter from parameter dictionary
        if hasattr(self,param):setattr(self,param,value)                        # if parameter present then sets its given value
        else:warnings.warn(f'Invalid paremeter -> {param} !',category=UserWarning)# else raise warning 
    else:raise ValueError(f'`params` must be dictionary of parameters, with key parameter name and value its value')# raise value error 

  
  # define method to get all paremeters 
  def get_params(self)->dict:

    ''' returns dictionary of all parameters '''

    return self.__dict__.copy()                                                 # make return all parameters dictionary copy 


#import tensorflow as tf                                                         # load tensorflow 
ckpt=UpdatedModelCheckpoint('/content/')                                        # make check point with updated class 
ckpt_params=ckpt.get_params()                                                   # get all parameters 
print('Parameters:',ckpt_params)                                                # see parameters dictionary
ckpt_params['save_best_only']=True                                              # update parameter "save_best_only" to True (default False)
ckpt_params['save_weights_only']=True                                           # update parameter "save_weights_only" to True (default False)
ckpt_params['epochs_since_last_save']=10                                        # update parameter "epochs_since_last_save" to 10 (default 0)
print('Current parameters:',ckpt.get_params())                                  # make print orginal paremetrs 
ckpt.set_params(ckpt_params)                                                    # try to make update parameters
print('Is same as updated?',ckpt_params==ckpt.__dict__)                         # make check both are same or not 
print('Updated parameters:',ckpt.get_params())                                  # updated parameters

import tensorflow as tf                                                         # load tensorflow 
import warnings                                                                 # load warnings 
class UpdatedEarlyStopping(tf.keras.callbacks.EarlyStopping):

  # define set parameters method 
  def set_params(self,params:dict)->None:

    ''' make set give parameters dictionary ''' 

    if isinstance(params,dict):                                                 # make check parameter must be dictionary 
      for param,value in params.items():                                        # get parameter from parameter dictionary
        if hasattr(self,param):setattr(self,param,value)                        # if parameter present then sets its given value
        else:warnings.warn(f'Invalid paremeter -> {param} !',category=UserWarning)# else raise warning 
    else:raise ValueError(f'`params` must be dictionary of parameters, with key parameter name and value its value')# raise value error 

  
  # define method to get all paremeters 
  def get_params(self)->dict:

    ''' returns dictionary of all parameters '''

    return self.__dict__.copy()                                                 # make return all parameters dictionary copy 

#import tensorflow as tf                                                         # load tensorflow 
lstp=UpdatedEarlyStopping()                                                     # make early stopping with updated class 
lstp_params=lstp.get_params()                                                   # get all parameters 
print('Parameters        :',lstp_params)                                        # see parameters dictionary
lstp_params['patience']=10                                                      # update parameter "patience" to 10 (default 0)
lstp_params['verbose']=1                                                        # update parameter "verbose" to 1 (default 0)
print('Updated parameters:',lstp_params)                                        # see updated parameters 
print('Object\'s current parameters:',lstp.get_params())                        # see parameters dictionary
lstp.set_params(lstp_params)                                                    # try to make update parameters
print('Object\'s updated parameters:',lstp.get_params())                        # see updated parameters dictionary
print('Is same as updated?',lstp_params==lstp.get_params())                     # make check both are same or not