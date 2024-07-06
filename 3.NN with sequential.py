import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train=x_train.reshape(-1,28*28).astype('float32')/255.0
x_test= x_test.reshape(-1,28*28).astype('float32')/255.0

#this keras sequential is convenient but less flexible
model=keras.Sequential([
    keras.Input(shape=(28*28)),
    layers.Dense(512,activation='relu'),
    layers.Dense(265,activation='relu'),
    layers.Dense(10)
    ])

# print(model.summary()) #to print this need to specify keras.Input
'''SparseCategoricalCrossentropy: when from_logits=True then output layer does not have activation func
but when fals, activation fucntion is needed'''
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=1)
model.evaluate(x_test,y_test,batch_size=32,verbose=1)



import sys
sys.exit()

##!
#**********************************************************************************************************
### functionalAPI keras =mulptiple input and multiple output
# we will add the name of the layers

'''as same as above'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train=x_train.reshape(-1,28*28).astype('float32')/255.0
x_test= x_test.reshape(-1,28*28).astype('float32')/255.0

'''modifications'''
inputs= keras.Input(shape=(784),name='initial')
x=layers.Dense(512,activation='relu',name='first_layer')(inputs)
x=layers.Dense(256,activation='relu',name='second_layer')(x)
out=layers.Dense(10,activation='softmax')(x)


model=keras.Model(inputs=inputs,outputs=out) #taking input and output to define a model

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer= keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)
print( model.summary())
model.fit(x_train,y_train,epochs=2,batch_size=32,verbose=1)
model.evaluate(x_test,y_test,batch_size=32,verbose=1)


##!
#***************************************************************************************************
#how to extract specific layer out or features of total model


'''as same as above'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
# print(x_train.shape)
x_train=x_train.reshape(-1,28*28).astype('float32')/255.0
x_test= x_test.reshape(-1,28*28).astype('float32')/255.0




'''modification'''
model1=keras.Sequential()
model1.add(keras.Input(shape=(28*28)))
model1.add(layers.Dense(512,activation='relu'))
model1.add(layers.Dense(265,activation='relu'))
model1.add(layers.Dense(128,activation='relu',name='my_layer1'))
model1.add(layers.Dense(10))



model11=keras.Model(inputs=model1.inputs,outputs=[model1.layers[-3].output])  ##-1=output, -2=last second,-3=last third

feature=model11.predict(x_train)
print(feature.shape,'third last layer')

#getting features by model mane
model2=keras.Model(inputs=model1.inputs,outputs=[model1.get_layer('my_layer1').output])
feature1=model2.predict(x_train)
print(feature1.shape,'my named layer')

#to get all features

model3=keras.Model(inputs=model1.inputs, outputs=[layer.output for layer in model1.layers])
features=model3.predict(x_train)
for i,feature in enumerate(features):
    print(feature.shape, 'layer{}'.format(i))