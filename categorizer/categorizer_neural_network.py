from __future__ import absolute_import, division, print_function, unicode_literals

import os
from time import time

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


root_path = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(root_path, 'data')

train_dir      = os.path.join(PATH, 'data_train')
validation_dir = os.path.join(PATH, 'data_validate')
models_dir     = os.path.join(PATH, '..', 'models')

train_ginger_dir    = os.path.join(train_dir, 'ginger')
train_asian_dir     = os.path.join(train_dir, 'asian')
train_mulatto_dir   = os.path.join(train_dir, 'mulatto')
train_brunette_dir  = os.path.join(train_dir, 'brunette')
train_dreadlock_dir = os.path.join(train_dir, 'dreadlock')

validation_ginger_dir    = os.path.join(validation_dir, 'ginger')
validation_asian_dir     = os.path.join(validation_dir, 'asian')
validation_mulatto_dir   = os.path.join(validation_dir, 'mulatto')
validation_brunette_dir  = os.path.join(validation_dir, 'brunette')
validation_dreadlock_dir = os.path.join(validation_dir, 'dreadlock')

num_ginger_train    = len(os.listdir(train_ginger_dir))
num_asian_train     = len(os.listdir(train_asian_dir))
num_mulatto_train   = len(os.listdir(train_mulatto_dir))
num_brunette_train  = len(os.listdir(train_brunette_dir))
num_dreadlock_train = len(os.listdir(train_dreadlock_dir))

num_ginger_validation    = len(os.listdir(validation_ginger_dir))
num_asian_validation     = len(os.listdir(validation_asian_dir))
num_mulatto_validation   = len(os.listdir(validation_mulatto_dir))
num_brunette_validation  = len(os.listdir(validation_brunette_dir))
num_dreadlock_validation = len(os.listdir(validation_dreadlock_dir))

total_train      = num_ginger_train      + num_asian_train      + num_mulatto_train      + num_brunette_train      + num_dreadlock_train
total_validation = num_ginger_validation + num_asian_validation + num_mulatto_validation + num_brunette_validation + num_dreadlock_validation

print('total training GINGER images:',    num_ginger_train)
print('total training ASIAN images:',     num_asian_train)
print('total training MULATTO images:',   num_mulatto_train)
print('total training BRUNETTE images:',  num_brunette_train)
print('total training DREADLOCK images:', num_dreadlock_train)

print('total validation GINGER images:',    num_ginger_validation)
print('total validation ASIAN images:',     num_asian_validation)
print('total validation MULATTO images:',   num_mulatto_validation)
print('total validation BRUNETTE images:',  num_brunette_validation)
print('total validation DREADLOCK images:', num_dreadlock_validation)

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
                                                           class_mode='categorical')


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              shuffle=True,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

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
    Conv2D(32, (3, 3),  padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(5, activation='softmax')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
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
model_path = os.path.join(models_dir, f'categorizer_model_{int(time())}.h5')
model.save(model_path)