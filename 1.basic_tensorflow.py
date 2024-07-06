##!
#to ignore tf gpu starting information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

physical_devices= tf.config.list_physical_devices('GPU')  # getting GPU access
tf.config.experimental.set_memory_growth(physical_devices[0],True)
exit()

'''' contents: 1.to create a tensor
               2.mathematical operations
               3.indexing
               4.reshaping'''

##!
# initilization of tensor
x=tf.constant(4,shape=(3,3),dtype= tf.float32) # creatinf 4X4 matirx
x1= tf.constant([[1,2,3],[2,3,4]],shape=(2,3)) #it is like a manual initialization
x3= tf.ones((3,3))
x4= tf.eye(3) #identity matirck

print(x4)

##!

## mathematical operation

x5= tf.random.normal((3,3),mean=0,stddev=1)
x6= tf.random.uniform((3,3),minval=0,maxval=2)
x7= tf.range(9)
x8= tf.cast(x7,dtype=tf.bool) # it converts datatype and bits use: tf.int,tf.bool
x9=tf.constant([1,2,3],shape=(3,),dtype=tf.float32)
x10=tf.constant([3,4,5],shape=(1,3),dtype=tf.float32)
x11=tf.matmul(x9, x10) #matrix multiplication
    #or
x11=x9@x10  #matrix multipliCATION

x12=tf.tensordot(x9,tf.transpose(x10),axes=0)
##!
###.....indexing......
x13=tf.constant([1,2,3,4,5,7])
print(x13[2:4])
print(x13[::2]) #skipping values
indices= tf.constant([3,2,5]) #inside array is indices value
x_ind=tf.gather(x13,indices) # values of the x13 according to the indices
print(x_ind)

##!

## reshaping...............................

p= tf.range(9)
print(p)
p1=tf.reshape(p,(3,3))
print(p1)
p2=tf.transpose(p1,perm=[1,0]) #perm= permutates the columns
print(p2)

##!
# function tf.tiles
import tensorflow as tf
x=tf.constant([[1,2,3]],dtype=tf.float32)
print(x)
a=tf.tile(x,tf.constant([3,1])) # creting multiple rows
print(a)
a1=tf.tile(x,tf.constant([1,2])) #creating multiple copies of columns
print(a1)












