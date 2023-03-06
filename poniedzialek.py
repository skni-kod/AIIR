import os

import tensorflow as tf
from model import *
import pathlib


def split_data(data_set, train_percent, val_percent, test_percent):
    data_set = data_set.shuffle(buffer_size=len(data_set), reshuffle_each_iteration=False)

    train_size = int(len(data_set) * train_percent)
    val_size = int(len(data_set) * val_percent)
    test_size = int(len(data_set) * test_percent)

    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size).take(val_size)
    test_ds = data_set.skip(train_size + val_size).take(test_size)

    return train_ds, val_ds, test_ds


def prepare_using_layers(ds, rot=False, bright=False, flip=False):
    """
    Function augments given dataset using Sequential model layers
    :param ds: dataset to be augmented
    :param rot: determines whether to apply random rotation
    :param bright: determines whether to apply random brightness
    :param flip: determines whether to appli random flips
    :return: augmented dataset
    """
    if rot:
        random_rot = tf.keras.Sequential([
            layers.RandomRotation(0.2),
        ])
        ds = ds.map(lambda x, y: (random_rot(x), y), num_parallel_calls=AUTOTUNE)
    if bright:
        random_bright = tf.keras.layers.RandomBrightness(factor=0.2)
        ds = ds.map(lambda x, y: (random_bright(x), y), num_parallel_calls=AUTOTUNE)
    if flip:
        random_flip = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
        ])
        ds = ds.map(lambda x, y: (random_flip(x), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    path = "datasets/gestures_dataset"  # Remember to recreate or change the path to the actual dataset
    data_dir = path

    batch_size = 32
    # (Mystyk) Image size might be different, just a guess based on webcam repo
    img_height = 300
    img_width = 300

    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        batch_size=batch_size,
        image_size=(img_height, img_width))

    AUTOTUNE = tf.data.AUTOTUNE

    # augumented_ds = prepare(data, shuffle=False, rot=True , rgb=True, brightness=False, saturation=True, hue=False)
    augmented_ds = prepare_using_layers(data, rot=True, bright=True, flip=True)

    train_ds, val_ds, test_ds = split_data(augmented_ds, 0.7, 0.2, 0.1)

    class_names = data.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    num_classes = len(class_names)

    epochs = 10

    functions = [create_model1, create_model2, create_model3, create_model4, create_model5]
    filters = [8, 16, 32]
    pool_size = [2, 4, 8]
    dense_units = [32, 64, 128, 256]
    counter = 1
    total = len(filters) * len(pool_size) * len(dense_units) * len(functions)
    for filter in filters:
        for pool in pool_size:
            for units in dense_units:
                for function in functions:
                    model = function(img_height, img_width, num_classes, filter, pool, units)
                    model.compile(optimizer='Adam',
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                  metrics=['accuracy'])

                    model.summary()
                    history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs
                    )

                    if not os.path.exists('models/'):
                        os.makedirs('models')
                    model.save('models/model_' + str(counter) + '.h5')
                    print(f"Progress: {counter}/{total}")
                    counter += 1
