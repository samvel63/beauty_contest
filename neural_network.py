from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import os
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt


PATH = '/home/samvel/projects/beauty_contest/data'

train_dir = os.path.join(PATH, 'data_train')
validation_dir = os.path.join(PATH, 'data_validate')


train_ginger_dir = os.path.join(train_dir, 'ginger')
train_asian_dir = os.path.join(train_dir, 'asian')
train_mulatto_dir = os.path.join(train_dir, 'mulatto')
# train_brunette_dir = os.path.join(train_dir, 'brunette')

validation_ginger_dir = os.path.join(validation_dir, 'ginger')
validation_asian_dir = os.path.join(validation_dir, 'asian')
validation_mulatto_dir = os.path.join(validation_dir, 'mulatto')
# validation_brunette_dir = os.path.join(validation_dir, 'brunette')

models_dir = os.path.join(PATH, 'models')

num_ginger_train = len(os.listdir(train_ginger_dir))
num_asian_train = len(os.listdir(train_asian_dir))
num_mulatto_train = len(os.listdir(train_mulatto_dir))
# num_brunette_train = len(os.listdir(train_brunette_dir))

num_ginger_validation = len(os.listdir(validation_ginger_dir))
num_asian_validation = len(os.listdir(validation_asian_dir))
num_mulatto_validation = len(os.listdir(validation_mulatto_dir))
# num_brunette_validation = len(os.listdir(validation_brunette_dir))

total_train = num_ginger_train + num_asian_train + num_mulatto_train
total_validation = num_ginger_validation + num_asian_validation + num_mulatto_validation

print('total training GINGER images:', num_ginger_train)
print('total training ASIAN images:', num_asian_train)
print('total training MULATTO images:', num_mulatto_train)
# print('total training BRUNETTE images:', num_brunette_train)

print('total validation GINGER images:', num_ginger_validation)
print('total validation ASIAN images:', num_asian_validation)
print('total validation MULATTO images:', num_mulatto_validation)
# print('total validation BRUNETTE images:', num_brunette_validation)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_validation)


batch_size = 128
epochs = 25
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
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

# model = Sequential([
#     Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     Dense(128, activation='relu'),
#     Dense(2, activation='softmax')
# ])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.summary()


if '-l' in sys.argv:
    path_index = sys.argv.index('-l') + 1
    model.load_weights(sys.argv[path_index])
else:
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

if '-s' in sys.argv:
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_dir = os.path.join(models_dir, f'model_{int(time())}.ckpt')
    model.save_weights(model_dir)

imgs_path = ['data_predict/unknow/1.jpg', 'data_predict/unknow/2.jpg']
imgs = []

for img_path in imgs_path:
    img_raw = tf.io.read_file(os.path.join(PATH, img_path))

    img_tensor = tf.image.decode_image(img_raw)
    img_tensor = tf.image.resize(img_tensor, [IMG_HEIGHT, IMG_WIDTH])
    img_tensor /= 255.0

    img = np.reshape(img_tensor, [1, IMG_HEIGHT, IMG_WIDTH, 3])
    imgs.append(img_tensor)

    classes = model.predict_classes(img)
    print(img_path, classes[0])
