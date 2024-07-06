##!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28*28).astype('float32')/255.0
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.0

##!
### Number 1.
'''a custom model using subclass, it is a very simple NN model'''
#
class My_model(keras.Model):
    def __init__(self, num_classes=10):
        super(My_model, self).__init__()
        self.dense1 = layers.Dense(64)
        self.dense2 = layers.Dense(num_classes)

    def call(self, input_tensor):
        x=tf.nn.relu(self.dense1(input_tensor))
        return self.dense2(x)

model=My_model()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=2,verbose=1)
# model.evaluate(x_test, y_test,verbose=1)

##!
## NUmber 2

'''what i want to do that to create a custom layer instead of layers.Dense'''

class Dense(layers.Layer):
    def __init__(self,units, input_dim):
        super().__init__()
        self.w= self.add_weight(
            name='w',
            shape= (input_dim,units),
            initializer='random_normal',
            trainable=True
        )
        self.b= self.add_weight(
            name='b',
            shape=(units,),
            initializer='zeros',
            trainable=True
        )
    def call(self,x1):
        return tf.matmul(x1,self.w)+self.b

class my_model(keras.Model):
    def __init__(self,num_classes=10):
        super().__init__()
        self.Dense1= Dense(64,784)
        self.Dense2= Dense(10,64)

    def call(self,in_tensor):
        x=tf.nn.relu(self.Dense1(in_tensor))
        return self.Dense2(x)


model1=my_model()
model1.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# model1.fit(x_train,y_train,epochs=1,verbose=1)
# model1.evaluate(x_test,y_test,verbose=1)


##!
## Number 3
'''now we want to remove input_dim from dense layer'''
class dense(keras.Model):
    def __init__(self,unit):
        super().__init__()
        self.unit=unit

    def build(self,input_dim): # this is a build function, which helps to omit input_shape arg in dense module
        self.w = self.add_weight(
            name='w',
            shape=(input_dim[-1], self.unit),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.unit,),
            initializer='zeros',
            trainable=True)

    def call (self,x2):
        return tf.matmul(x2,self.w)+ self.b

class my_relu(layers.Layer): # my edited relu
    def __init__(self):
        super().__init__()

    def call(self,x):
        return tf.math.maximum(x,0)



class mymodel(keras.Model):
    def __init__(self,num_classes=10):
        super().__init__()
        self.dense1=dense(64)
        self.dense2=dense(num_classes)
        self.relu=my_relu()                 # defining my_relu

    def call(self, x3):
        x=self.relu(self.dense1(x3))
        return self.dense2(x)

model4=mymodel()
model4.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model4.fit(x_train,y_train,epochs=2,verbose=1)
model4.evaluate(x_test,y_test,verbose=1)