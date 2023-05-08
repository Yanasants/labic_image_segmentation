from keras import *
from keras.models import Model
from keras.layers import *

class UnetOriginal:
    def __init__(self, input_layer_shape):
        self.input_layer_shape = input_layer_shape

    def generate_model(self):
        input_layer = Input(shape=self.input_layer_shape) 

        c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
        l = MaxPool2D(strides=(2,2))(c1) 
        c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
        l = MaxPool2D(strides=(2,2))(c2)
        c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
        l = MaxPool2D(strides=(2,2))(c3)
        c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)

        l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
        l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
        l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
        l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
        l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
        l = Dropout(0.5)(l)

        output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l) 
                                                                
        model = Model(input_layer, output_layer)
        return model

