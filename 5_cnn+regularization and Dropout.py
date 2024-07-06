# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers,regularizers

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

def my_model():
    initial=keras.Input(shape=(32,32,3))
    x=layers.Conv2D(32,3,padding='same',kernel_regularizer= regularizers.l2(0.01))(initial)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,3,padding='same',kernel_regularizer= regularizers.l2(0.01))(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(128,3,padding='same',kernel_regularizer= regularizers.l2(0.01))(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu', kernel_regularizer= regularizers.l2(0.01))(x)
    x=layers.Dropout(0.5)(x) # 50% connection will be omitting bt upper and lower layers
    out=layers.Dense(10)(x)
    model= keras.Model(inputs=initial,outputs=out)

    return model



model1= my_model()

model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.Adam(lr=3e-4),
    metrics= ['accuracy']
)

model1.fit(x_train,y_train,epochs=4,batch_size=32,verbose=1)

model1.evaluate(x_test,y_test, batch_size=32,verbose=1)


