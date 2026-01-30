import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,batch_size,dataframe,x_col,y_col,number_classes,dimensions=[256,1600,3],shuffle=True):
        self.batch=batch_size
        self.df=dataframe
        self.x=x_col
        self.y=y_col
        self.classes=number_classes
        self.dim=dimensions
        self.indexes=self.df.index.tolist()
        self.shuffle=shuffle
        self.index_of_indexes=np.arange(len(self.indexes))
        self.on_epoch_end()
        self.n=0
        self.max=self.__len__()
    def __len__(self):
        return int(np.floor(len(self.indexes)/self.batch))
    def on_epoch_end(self):
        if self.shuffle==True:
            np.random.shuffle(self.index_of_indexes)
    def __next__(self):
      if self.n>=self.max:
        self.n=0
      result = self.__getitem__(self.n)
      self.n += 1
      return result
    def __getitem__(self,index):
        temp_index_of_indexes=self.index_of_indexes[index*self.batch:(index+1)*self.batch]
        temp_indexes=[self.indexes[i] for i in temp_index_of_indexes]
        X=np.empty((self.batch,self.dim[0],self.dim[1],self.dim[2]))
        Y=np.empty((self.batch,self.dim[0],self.dim[1],self.classes))
        for i,id_ in enumerate(temp_indexes):
            image_name=str(self.df.loc[id_,self.x])
            classes_list=np.array(self.df.loc[id_,self.y])
            shape=[self.dim[0],self.dim[1]]
            X[i,],Y[i,]=self.get_data(image_name,classes_list,shape)
        return X,Y
    def get_data(self,image_name,classes_list,shape):
        for i,c in enumerate(classes_list):
            if i==0 and c==1:
                file=image_name.split('.')[0]+'.npy'
                path='/content/severstal-steel-defect-detection/temp1/'+file
                channel1=np.load(path)
                channel1=channel1/255.0
            elif i==0 and c==0:
                channel1=np.zeros((shape[0],shape[1]))
            elif i==1 and c==1:
                file=image_name.split('.')[0]+'.npy'
                path='/content/severstal-steel-defect-detection/temp2/'+file
                channel2=np.load(path)
                channel2=channel2/255.0
            elif i==1 and c==0:
                channel2=np.zeros((shape[0],shape[1]))
            elif i==2 and c==1:
                file=image_name.split('.')[0]+'.npy'
                path='/content/severstal-steel-defect-detection/temp3/'+file
                channel3=np.load(path)
                channel3=channel3/255.0
            elif i==2 and c==0:
                channel3=np.zeros((shape[0],shape[1]))
            elif i==3 and c==1:
                file=image_name.split('.')[0]+'.npy'
                path='/content/severstal-steel-defect-detection/temp4/'+file
                channel4=np.load(path)
                channel4=channel4/255.0
            elif i==3 and c==0:
                channel4=np.zeros((shape[0],shape[1]))
        path='/content/severstal-steel-defect-detection/train_images/'+image_name
        image=load_img(path,target_size=(shape[0],shape[1],3))
        image=img_to_array(image)
        image=image/255.0
        mask=np.stack([channel1,channel2,channel3,channel4],axis=-1)
        image=tf.cast(image,dtype=tf.float32)
        mask=tf.cast(mask,dtype=tf.float32)  
        return image,mask

batch1=4 * tpu_strategy.num_replicas_in_sync
batch2=2 * tpu_strategy.num_replicas_in_sync


with tpu_strategy.scope():
  training_model=custom_model()
  def soft_dice_loss(y_true,pred):
    y_true=K.flatten(y_true)
    pred=K.flatten(pred)
    intersec=(2*K.sum(y_true*pred))+1e-9
    deno=K.sum(y_true**2)+K.sum(pred**2)+1e-9
    return 1-K.mean(intersec/deno)
  def soft_dice_coeff(y_true,pred):
    y_true=K.flatten(y_true)
    pred=K.flatten(pred)
    intersec=(2*K.sum(y_true*pred))+1e-9
    deno=K.sum(K.abs(y_true))+K.sum(K.abs(pred))+1e-9
    return K.mean(intersec/deno)
  training_model.compile(
      optimizer='Adam',
      loss=soft_dice_loss,
      metrics=[soft_dice_coeff])