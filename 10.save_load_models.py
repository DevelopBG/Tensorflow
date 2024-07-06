##!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(-1, 28*28).astype('float32')/255.0
x_test = x_test.reshape(-1, 28*28).astype('float32')/255.0
###****************************************************************************************************
#1. how to save and load model weights
#2. How to save and load entire model ( called serializing model)
#        - save the weights
#        - save the model architecgture
#        - training configuration (model.compile())
#        - save optimizer and states



model1=keras.Sequential([
    keras.Input(shape=(28*28)),
    layers.Dense(512,activation='relu'),
    layers.Dense(265,activation='relu'),
    layers.Dense(10)
    ])

inputs= keras.Input(784)
x= layers.Dense(64,activation='relu')(inputs)
out=layers.Dense(10)(x)
model2=keras.Model(inputs=inputs,outputs=out)

class mymodel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1=layers.Dense(64,activation='relu')
        self.dense2=layers.Dense(10)
    def call(self,x):
        x1=tf.nn.relu(self.dense1(x))
        return self.dense2(x1)

model3=mymodel()

model_dir='D:\PYTHON_PROJECTS/tensorflow/saved_weights'

model=model2
model.load_weights(model_dir)   # loading the saved weights only

model.compile(
    optimizer= keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# model.fit(x_train,y_train,epochs=2,verbose=1)
# model.evaluate(x_test,y_test,verbose=1)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_weights(model_dir ) # only saving the weights
##************************************************
#saving full model
model.save('full_model')

#As the whole model was saved, so no need to run the whole one again
model=keras.models.load_model('full_model')
model.fit(x_train,y_train,batch_size=32,epochs=2,verbose=1)
model.evaluate(x_test,y_test,verbose=1)




