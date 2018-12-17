import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

# Number of filters at different stages
nf1 = 32
nf2 = 2*nf1
nf3 = 2*nf2
nf4 = 2*nf3
nf5 = 2*nf4

def segnet(pretrained_weights = None,input_size = (512, 512, 1)):
    X = Input(input_size)
    
    #Stage1
    conv1 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(X)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1, mask1 = MaxPoolingWithArgmax2D((2, 2))(conv1)
    
    #Stage2
    conv2 = Conv2D(nf2, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(nf2, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2, mask2 = MaxPoolingWithArgmax2D((2, 2))(conv2)
    
    #Stage3
    conv3 = Conv2D(nf3, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(nf3, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3, mask3 = MaxPoolingWithArgmax2D((2, 2))(conv3)
    
    #Stage4
    conv4 = Conv2D(nf4, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(nf4, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.5)(conv4)
    #drop4 = Dropout(0.2)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4, mask4 = MaxPoolingWithArgmax2D((2, 2))(conv4)
    
    #Stage5
    conv5 = Conv2D(nf4, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(nf4, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    #conv5 = Dropout(0.5)(conv5)
    #drop5 = Dropout(0.2)(conv5)
    
    #Stage6
    up6 = Conv2D(nf3, kernel_size=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(MaxUnpooling2D((2, 2))([conv5, mask4]))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    #merge6 = concatenate([drop4,up6], axis=3)
    conv6 = Conv2D(nf3, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(nf3, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    #Stage7
    up7 = Conv2D(nf2, kernel_size=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(MaxUnpooling2D((2, 2))([conv6, mask3]))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    #merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(nf2, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(nf2, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    #Stage8
    up8 = Conv2D(nf1, kernel_size=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(MaxUnpooling2D((2, 2))([conv7, mask2]))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    #merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    #Stage9
    up9 = Conv2D(nf1, kernel_size=(2, 2), padding = 'same', kernel_initializer = 'he_normal')(MaxUnpooling2D((2, 2))([conv8, mask1]))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    #merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(nf1, kernel_size=(3, 3), padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    #Stage10
    conv10 = Conv2D(1, kernel_size=(1, 1), activation = 'sigmoid')(conv9)

    model = Model(inputs = X, outputs = conv10)
    #model.compile(optimizer = Adam(lr = 1e-5), loss = iou_coef_loss, metrics = [iou_coef])
    #model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])
    model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
