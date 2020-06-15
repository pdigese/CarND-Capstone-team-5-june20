
from keras.models import Sequential, load_model 
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D 
from keras.layers import Conv2D, Cropping2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint



def Model_tl():
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(100,100,3)))
    model.add(Conv2D(filters=3,kernel_size = (1,1),strides=(1, 1),padding="valid"))
    
    model.add(Conv2D(filters=32,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(Conv2D(filters=32,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(Conv2D(filters=64,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(Conv2D(filters=128,kernel_size = (3,3),strides=(1, 1),padding="valid"))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    return model
