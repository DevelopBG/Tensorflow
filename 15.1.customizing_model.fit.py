''''in 15.1===here we will do custom training,
 in 15.2== custom compile'''''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train= x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test= x_test.reshape(-1,28,28,1).astype('float32')/255.0

model= keras.Sequential([
    keras.Input((28,28,1)),
    layers.Conv2D(16,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)

])

class CustomFit(keras.Model):
    def __init__(self,model):
        super().__init__()
        self.model= model

    def train_step(self,data):
        x,y= data # tuple
        # now doing forward propagation and loss function.we will be doing this under that 'tape'.
        # it records the steps and values for backpropagation

        with tf.GradientTape() as tape:
            # forward propagation
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)  # this operation will be done in compile section




        training_variable= self.trainable_variables
        gradient= tape.gradient(loss, training_variable ) #obtaining gradient of loss w.r.t training_variable

        self.optimizer.apply_gradients(zip(gradient, training_variable))
        self.compiled_metrics.update_state(y,y_pred) # this is for accuracy

        return {m._name : m.result() for m in self.metrics} # m.name= accuracy


training=CustomFit(model)
training.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics= ['accuracy'])

training.fit(x_train,y_train, batch_size=32, epochs=2)