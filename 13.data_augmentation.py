import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train,ds_test),ds_info= tfds.load(
    'cifar10',
    split=['train','test'],
    shuffle_files= True,
    as_supervised= True,
    with_info=True
)

def normalize_img(img,label):
    #image normalization
    return tf.cast(img, tf.float32)/255.0 , label

AUTOTUNE= tf.data.experimental.AUTOTUNE
BATCH_SIZE=32

# dataset augmentation

def augment(img,label):
    new_height=new_width=32
    img=tf.image.resize(img,(new_height,new_width)) #resizing images
    #now we do not want all images in RGB, some in gray.
    if tf.random.uniform((),minval=0,maxval=1)<0.1:
        img=tf.tile(tf.image.rgb_to_grayscale(img),[1,1,3]) # as we change images channels,so we need to match the channel names in Conv layers
                                                            # in short img_gray has channel 1 anf model will expect 3 no of channels,
                                                            # here third layer has been copied 3 times

    img=tf.image.random_brightness(img,max_delta=0.1)       #adding random brightness to the images
    img=tf.image.random_contrast(img,lower=0.1,upper=0.2)
    img=tf.image.random_flip_left_right(img)

    return img,label
                                    #OR
'''on the other hand , this augmentation can be done inside the sequential model,
it is layers.experimental.preprocessing'''

data_augmentation= keras.Sequential([
    layers.experimental.preprocessing.Resizing(height=32,width=32),
    layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
    layers.experimental.preprocessing.RandomContrast(factor=0.1)
])



# setup training date
ds_train= ds_train.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.cache()
ds_train=ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train=ds_train.map(augment,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(AUTOTUNE)
#setup of testing dataset

ds_test=ds_test.map(normalize_img,num_parallel_calls=AUTOTUNE)
ds_test=ds_test.cache()
ds_test=ds_test.shuffle(ds_info.splits['test'].num_examples)
ds_test=ds_test.batch(BATCH_SIZE)
ds_test=ds_test.prefetch(AUTOTUNE)




model=keras.Sequential([
    keras.Input(shape=(32,32,3)),
    data_augmentation, #we need to add above sequential data_augmentation
    layers.Conv2D(12,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
])


model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(ds_train,epochs=4,verbose=1)
model.evaluate(ds_test,verbose=1)

