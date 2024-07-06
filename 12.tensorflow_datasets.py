##!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

''' tensorflow has hige datasets unlike keras. So we need to learn how to work with those datasets'''
physical_devices= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
##!
(ds_train,ds_test),ds_info= tfds.load(
    'mnist',
    split=['train','test'], #mnist does not have validation set
    shuffle_files=True,
    as_supervised=True,     #it returns tuple (image,label), otherwise (if false) returns dictionary
    with_info= True,        #for ds_info
)

# print(ds_info)
# fig=tfds.show_examples(ds_train,ds_info,rows=4,cols=4) # as_supervised= False must, it shows the datas.
BATCH_SIZE=63
def normalize_img(image,label):
    return tf.cast(image,tf.float32)/255.0 , label

AUTOTUNE= tf.data.experimental.AUTOTUNE
ds_train=ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE) #num_parallel...= it sets the how many
                                                                 # parallel call needs to set, autotune is
                                                                 #set by tensorflow, otherwise we can set any number like 2,3,4,5 etc.
ds_train=ds_train.cache()               #keep dataset in the memory to load faster after first time
# ds_train=ds_train.shuffle(1000)         #1000=shuffle buffer size.
                    #or
ds_train=ds_train.shuffle( ds_info.splits['train'].num_examples)  #it also shuffle the whole training set randomly
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)    # it prefetch 64 example when GPU is called to run the code faster

ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(128)
ds_test=ds_test.prefetch(AUTOTUNE)

#as dataset has been loaded, now define a model and test the model using the dataset

model=keras.Sequential([
    keras.Input(shape=(28,28,1)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(31,3,activation='relu'),
    layers.Flatten(),
    layers.Dense(10)
])

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(ds_train,epochs=6,verbose=1)
model.evaluate(ds_test,verbose=1)


##!
'''lets focus on dataset, here work has been done on imdb dataset,
it actually work of NLP. So I have skipped it. Later it can be learned, look forward'''

