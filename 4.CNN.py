##!

#primary loading dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

##!

'''basic model, later we will define advance model'''

model=keras.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(16,(3,3),padding='valid',activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(), #default is (2,2)
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
])
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy']
)

# model.fit(x_train,y_train,batch_size=63,epochs=1,verbose=2)

# model.evaluate(x_test,y_test,batch_size=63,verbose=2)


##!

#funtional API

def my_model():
    inputs=keras.Input(shape=(32,32,3)) #since we use Batchnormalization, do not put activation in Conv2D layer
    x=layers.Conv2D(32,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,5,padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu')(x)
    out=layers.Dense(10)(x)
    model=keras.Model(inputs=inputs,outputs=out)

    return model
model1=my_model()
model1.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.Adam(lr=3e-4),
    metrics=['accuracy']
    )

model1.fit(x_train,y_train,batch_size=64,epochs=4,verbose=1)
model1.evaluate(x_test,y_test,batch_size=64,verbose=1)



