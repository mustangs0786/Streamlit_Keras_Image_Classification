import os
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
######################### paramters ##########

CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 2
BATCH_SIZE_TRAINING = 30
BATCH_SIZE_VALIDATION = 20
BATCH_SIZE_TESTING = 1
split_at = 140

####################################
def neural_model(NUM_CLASSES):
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))
    
    model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
    # for layer in model.layers[:split_at]: layer.trainable = False
    # for layer in model.layers[split_at:]: layer.trainable = True
    model.layers[0].trainable = False
    print('entered model')
    # print(model)
    return model


def train_reading_data(cwd):
    # st.text(os.path.join(cwd+'/'+str('train')))
    tain_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, \
                            shear_range=0.2, zoom_range=0.2)
    validation_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = tain_data_generator.flow_from_directory(
            os.path.join(cwd+'/data/'+str('train')),
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            batch_size=BATCH_SIZE_TRAINING,
            class_mode='categorical')
    validation_generator = validation_data_generator.flow_from_directory(
            os.path.join(cwd+'/data/'+str('validation')),
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            batch_size=BATCH_SIZE_VALIDATION,
            class_mode='categorical') 
    
    label_map = dict((v,k) for k,v in train_generator.class_indices.items())
    # st.text(label_map)
    return train_generator,validation_generator,label_map


def model_training(cwd,train_generator,validation_generator,NUM_CLASSES,train_data_length,validation_data_length):
    # st.text(f'{cwd}/best.hdf5')
    STEPS_PER_EPOCH_TRAINING = int( np.ceil(train_data_length / BATCH_SIZE_TRAINING) )
    STEPS_PER_EPOCH_VALIDATION = int( np.ceil(validation_data_length / BATCH_SIZE_VALIDATION) )
    # st.text(STEPS_PER_EPOCH_TRAINING)
    # st.text(STEPS_PER_EPOCH_VALIDATION)
    cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
    cb_checkpointer = ModelCheckpoint(filepath = f'{cwd}/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')
    model = neural_model(NUM_CLASSES)
    # print(model)
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)
    fit_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        callbacks=[cb_checkpointer, cb_early_stopper])
    st.line_chart(fit_history.history)
    # fit_history.to_csv('fit.csv',as_index=False)
# model.load_weights("../working/best.hdf5")    

def test_reading_data(cwd):
    # st.text(os.path.join(cwd+'/'+str('test')))
    test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_data_generator.flow_from_directory(
            os.path.join(cwd+'/data/'+str('test')),
            target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
            batch_size=BATCH_SIZE_TESTING,
            shuffle=False
            )
    return test_generator


def model_prediction(cwd,test_generator):
    model = tf.keras.models.load_model(os.path.join(cwd+'/'+str('best.hdf5')))
    predict = model.predict_generator(test_generator)
    result = np.argmax(predict, axis=-1)
    # st.text(result)
    return result





