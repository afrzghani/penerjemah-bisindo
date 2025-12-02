import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(glob('gestures/*'))

image_x, image_y = get_image_size()

def cnn_model_improved():
    num_of_classes = get_num_of_classes()
    
    model = Sequential()
    
    model.add(Conv2D(8, (3,3), input_shape=(image_x, image_y, 1), 
    activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(16, (3,3), activation='relu', 
    kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(num_of_classes, activation='softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    filepath = "cnn_model_keras2.h5"
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
    save_best_only=True, mode='max')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, 
    restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
    patience=3, min_lr=0.00001, verbose=1)
    
    callbacks_list = [checkpoint, early_stop, reduce_lr]
    return model, callbacks_list

def train_with_augmentation():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    print(f"Training data shape: {train_images.shape}")
    print(f"Validation data shape: {val_images.shape}")
    print(f"Validation labels shape: {val_labels.shape}")

    model, callbacks_list = cnn_model_improved()
    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=64),
        validation_data=(val_images, val_labels),
        epochs=15,
        steps_per_epoch=len(train_images) // 64,
        callbacks=callbacks_list,
        verbose=1
    )
    
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("\n" + "="*50)
    print(f"Validation Accuracy: {scores[1]*100:.2f}%")
    print(f"Validation Error: {(100-scores[1]*100):.2f}%")
    print("="*50)
    
    return model, history

model, history = train_with_augmentation()
K.clear_session()
