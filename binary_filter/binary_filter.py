from __future__ import absolute_import, division, print_function, unicode_literals

import os
from time import time

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


root_path = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(root_path, 'data')

train_dir      = os.path.join(PATH, 'data_train')
validation_dir = os.path.join(PATH, 'data_validate')
models_dir     = os.path.join(PATH, '..', 'models')

train_girl_dir  = os.path.join(train_dir, 'girl')
train_lit_dir   = os.path.join(train_dir, 'lit')

validation_girl_dir = os.path.join(validation_dir, 'girl')
validation_lit_dir  = os.path.join(validation_dir, 'lit')


num_girl_train = len(os.listdir(train_girl_dir))
num_lit_train  = len(os.listdir(train_lit_dir))

num_girl_validation = len(os.listdir(validation_girl_dir))
num_lit_validation  = len(os.listdir(validation_lit_dir))


total_train      = num_girl_train      + num_lit_train     
total_validation = num_girl_validation + num_lit_validation 

print('total training girl images:', num_girl_train)
print('total training lit images:',  num_lit_train)

print('total validation girl images:', num_girl_validation)
print('total validation lit images:',  num_lit_validation)

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_validation)


batch_size = 128
epochs = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=.2,
                    height_shift_range=.2,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

validation_image_generator = ImageDataGenerator(rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              shuffle=True,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, classers = next(train_data_gen)


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_images(sample_training_images[:5])
print(classers[:5])


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_validation // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save FULL model
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model_path = os.path.join(models_dir, f'binary_filter_model_{int(time())}.h5')
model.save(model_path)
