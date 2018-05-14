# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# from keras.layers import Dense, Activation, Dropout, Flatten
# from keras import optimizers
# from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
# import numpy as np

from model import load_model

IMG_SIZE = 150
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 16

TRAIN_DIR = './data/train'
TEST_DIR = './data/test'

# Step 1: Load Data
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    class_mode='binary',
    batch_size=BATCH_SIZE)

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    classes=['dogs', 'cats'],
    class_mode='binary',
    batch_size=BATCH_SIZE)

# Step 2: Import and Setup Model
model = load_model(IMG_SIZE)

# Step 3: Train Model
training = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=2048 // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=832 // BATCH_SIZE)

# model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1)

# Step 4: Save Model
model.save_weights('models/CNN.h5')
