import tensorflow as tf
from tensorflow import keras

new_input = Input(shape=(128 , 128 , 3))
base = EfficientNetB0(include_top=False ,
                input_tensor=new_input )

for layer in base.layers :
    layer.trainable = False
    
x = base.output
x = Conv2D (64 , (3,3) , activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = GlobalAveragePooling2D ()(x)

out = Dense (6 , activation = 'softmax')(x)

model = Model (inputs = base.input , outputs = out)

cce = tf.keras.losses.CategoricalCrossentropy()

model.compile(loss=cce,
              optimizer=Adam(),
              metrics=kappa_score)

history = model.fit_generator(train_generator,
                    validation_data=val_generator, 
                    epochs = 100)