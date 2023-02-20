import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from keras import layers
from keras import Sequential
import pathlib


def split_data(data_set, train_percent, val_percent, test_percent):

    data_set = data_set.shuffle(buffer_size=len(data_set), reshuffle_each_iteration=False)

    train_size = int(len(data_set)*train_percent)
    val_size = int(len(data_set)*val_percent)
    test_size = int(len(data_set)*test_percent)

    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size).take(val_size)
    test_ds = data_set.skip(train_size+val_size).take(test_size)

    return train_ds, val_ds, test_ds


def image_rot_on_dataset(image, label):
    image = tf.image.rot90(image)
    return image, label


def random_brightness(image, label):
    augment_image = tf.image.random_brightness(image, 1.0)
    return augment_image, label


def random_hue(image, label):
    hue_image = tf.image.random_hue(image, 0.5)
    return hue_image, label


def random_saturation(image, label):
    saturated_image = tf.image.random_saturation(image, 0, 20)
    return saturated_image, label


def rgb_to_gray_scale(image, label):
    gray_scale_image = tf.image.rgb_to_grayscale(image)
    return gray_scale_image, label


def prepare(ds, shuffle=False, rot=False, rgb=False, brightness=False, saturation=False, hue=False):

    if shuffle:
        ds = ds.shuffle(1000)

    if saturation:
        ds = (
            ds.map(random_saturation, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    if brightness:
        ds = (
            ds.map(random_brightness, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    if rgb:
        ds = (
            ds.map(rgb_to_gray_scale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    if rot:
        ds = (
            ds.map(image_rot_on_dataset, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    if hue:
        ds = (
            ds.map(random_hue, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )

    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':

    path = "C:/Users/Lenovo/Downloads/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=path, untar=True)
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 180
    img_width = 180

    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        batch_size=batch_size,
        image_size=(img_height, img_width))

    train_ds, val_ds, test_ds = split_data(data, 0.7, 0.2, 0.1)

    ## wyciągnięcie jednego elementu ze zbioru train
    train_one_example = train_ds.take(1)

    # stworzenie nowego zbioru danych z jednym elementem
    one_example_dataset = tf.data.Dataset.from_tensors(train_one_example)

    AUTOTUNE = tf.data.AUTOTUNE

    ds = prepare(one_example_dataset, shuffle=False, rot=False, rgb=True, brightness=False, saturation=False, hue=False)

    class_names = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    num_classes = len(class_names)

    #  Here data preparation has ended

    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    #  Here model creation has ended

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
