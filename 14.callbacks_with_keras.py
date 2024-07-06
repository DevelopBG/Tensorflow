'''MAIN INTEREST: WE HAVE SAVE MOdel previously. But here we want to save model every epoch and
want to save best model'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from tensorflow import keras

physical_device= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0],True)
#load dataset
#normalised
#autotune
#batch_size
#dataset sugmentation

(ds_train,ds_test),ds_info= tfds.load('mnist',split=['train','test'],with_info=True,shuffle_files= True,
                                      as_supervised=True)

def normalize_img(img,label):
    return tf.cast(img,tf.float32)/255.0, label

AUTOTUNE= tf.data.experimental.AUTOTUNE
BATCH_SIZE=128

def augment(img,label):
    new_height=new_width=32
    img=tf.image.resize(img,(new_width,new_height))
    img=tf.image.random_brightness(img,max_delta=0.8)
    img=tf.image.random_flip_left_right(img)
    return img, label

# training dataset preparation
ds_train=ds_train.map(normalize_img, num_parallel_calls= AUTOTUNE)
ds_train=ds_train.cache()
ds_train= ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train= ds_train.map(augment,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)

# testing dataset preparation

ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.cache()
ds_test=ds_test.shuffle(ds_info.splits['test'].num_examples)
ds_test=ds_test.map(augment, num_parallel_calls=AUTOTUNE)
ds_test=ds_test.batch(BATCH_SIZE)
ds_test=ds_test.prefetch(AUTOTUNE)

model= keras.Sequential([
    keras.Input(shape=(32,32,1)),
    layers.Conv2D(16,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(120,activation='relu'),
    layers.Dense(10)

])

# callback method
# keras has bunch of different callback. check the website to explore more
# model will be saved at the root directory where code is stored
save_callback= keras.callbacks.ModelCheckpoint(
    'checkpoints/',
    save_weights_only=True,
    monitor='accuracy',
    sae_best_only=False
)


model.compile(optimizer=keras.optimizers.Adam(lr=3e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
# changing learning rate as model progresses
def scheduler(epoch,lr):
    if epoch <2:
        return lr
    else:
        return lr * 0.99

lr_scheduler= keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

model.fit(ds_train,
          epochs=5,
          verbose=1,
          callbacks=[save_callback,lr_scheduler])
# model.evaluate(ds_test,verbose=1)