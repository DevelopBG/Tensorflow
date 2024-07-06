##!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
device_name = tf.test.gpu_device_name()

# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

(x_train,y_train),(x_test,y_test)=mnist.load_data()
# print(x_train.shape)
x_train=x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test=x_test.reshape(-1,28,28,1).astype('float32')/255.0
# print(x_train.shape)
##!
# #CNN->batchnorm-> Relu(common structure


class CNNBlock(layers.Layer):

    def __init__(self,out_channel,kernel_size=3):
        super().__init__()
        self.conv=layers.Conv2D(out_channel,kernel_size,padding='same')
        self.bn=layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x=self.conv(input_tensor)
        x=self.bn(x,training=training)
        x=tf.nn.relu(x)
        return x

#**************************************************************************************
'''simpler one'''
# model=keras.Sequential([
#         CNNBlock(32),
#         CNNBlock(64),
#         layers.Flatten(),
#         layers.Dense(10)
#
#     ]
# )
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )
# model.fit(x_train,y_train,epochs=2,verbose=2)
# model.evaluate(x_test,y_test,verbose=2)
##!
#**********************************************************************************************
'''complicated one'''

class ResBlock(layers.Layer):
    def __init__(self,channels):
        super().__init__()
        self.channels=channels
        self.cnn1=CNNBlock(channels[0])
        self.cnn2=CNNBlock(channels[1])
        self.cnn3=CNNBlock(channels[2])
        self.pooling=layers.MaxPooling2D()
        self.identity_mapping=layers.Conv2D(channels[1],kernel_size=3, padding='same')

    def call(self,input_tensor,training=False):
        x=self.cnn1(input_tensor,training=training)
        x=self.cnn2(x,training=training)
        x=self.cnn3(x+ self.identity_mapping(input_tensor),training=training)
        return self.pooling(x)


class ResNet_like(keras.Model):
    def __init__(self,num_classes=10):
        super().__init__()
        self.block1=ResBlock([32,32,64])
        self.block2=ResBlock([128,128,256])
        self.block3=ResBlock([128,256,512])
        self.pool=layers.GlobalAveragePooling2D() # this somewhat works like layers.Flatten()
        self.classifier=layers.Dense(num_classes)

    def call(self,input_tensor, training=False):
        x=self.block1(input_tensor,training=training)
        x=self.block2(x,training=training)
        x=self.block3(x,training=training)
        x=self.pool(x,training=training)
        # print(x.shape)
        return self.classifier(x)

    def model(self):
        x=keras.Input(shape=(28,28,1))
        return keras.Model(inputs=[x], outputs=self.call(x))

model = ResNet_like(num_classes=10)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=1,batch_size=64,verbose=1) # when we call model.fit,it internally makes training=True
print(model.model().summary())
model.evaluate(x_test,y_test,verbose=1) # here internally training=False







