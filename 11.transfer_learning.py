##!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

physical_device=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0],True)
##!
#-------------------------------------------------------------------#
#                       Pretrained model
#___________________________________________________________________#

(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test=x_test.reshape(-1,28,28,1).astype('float32')/255.0

model= keras.models.load_model('pretrained') # pretrained is an older model, taken from net or my own work


# print(model.summary())
'''now I want to remove last layer and replace by my own modified layer'''
base_input = model.layers[0].input
base_output = model.layers[-2].output
out = layers.Dense(10)(base_output)                                 #changed last layer
model1 = keras.Model(inputs=base_input,outputs=out)

# print(model1.summary())
model1.compile(
    optimizer=keras.optimizers.Adam(),
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']

    )
# model1.fit(x_train,y_train,epochs=2,verbose=1)
# model1.evaluate(x_test,y_test,verbose=1)
##!
#*******************************************************************************************************
### how to freeze train model's layer/layers
model2= keras.models.load_model('pretrained')
'''it freezes all layer from training'''
model2.trainable= False
#               or
# for layer in model2.layers[0:4]:
#     assert layer.trainable==False
#     layer.trainable=False



# print(model2.summary())
'''now I want to remove last layer and replace by my own modified layer'''
base_input = model2.layers[0].input
base_output = model2.layers[-2].output
out = layers.Dense(10)(base_output)                                 #changed last layer
model3 = keras.Model(inputs=base_input,outputs=out)

# print(model3.summary())
model3.compile(
    optimizer=keras.optimizers.Adam(),
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']

    )
# model3.fit(x_train,y_train,epochs=2,verbose=1)
# model3.evaluate(x_test,y_test,verbose=1)
'''freezing a layers of pretrained model improves training time, and sometime for gigantic 'we could freeze some 
layers from training and then train some modified layers as our need'''
##!
#--------------------------------------------------------------------#
#                       Pretrained Keras model
#____________________________________________________________________#
'''here we will use keras pretrained models and apply custom made dataset'''
x=tf.random.normal(shape=(5,299,299,3))  #Randomly generate 5 images of size (299x299) of RGB
y=tf.constant([0,1,2,3,4])              # the labels of 5 generated images

model4= keras.applications.InceptionV3(include_top=True)
# print(model4.summary())

base_input= model4.layers[0].input
base_out=model4.layers[-2].output
final_out=layers.Dense(5)(base_out)
new_model=keras.Model(inputs=base_input,outputs=final_out)
new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
new_model.fit(x,y,epochs=4,verbose=1)










#----------------------------------------------------------------------#
#                        Pretrained Hub Model
#----------------------------------------------------------------------#

#https://tfhub.dev/
x=tf.random.normal(shape=(5,299,299,3))
y=tf.constant([0,1,2,3,4])
url='https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'
base_model=hub.KerasLayer(url,input_shape=(299,299,3))
base_model.trainable=False # it can be done, as i do not want to train those layers
model=keras.Sequential([
    base_model,
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(5)
])
# model.compile(.....)
# model.fit(...)

