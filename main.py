import matplotlib.pyplot as plt
import tensorflow as tf
import json
from keras import layers
import keras_tuner as kt
import pathlib
from hyperModel import create_HyperModel
from keras import layers
from keras import Sequential


def split_data(data_set, train_percent, val_percent, test_percent):
    data_set = data_set.shuffle(buffer_size=len(data_set), reshuffle_each_iteration=False)

    train_size = int(len(data_set) * train_percent)
    val_size = int(len(data_set) * val_percent)
    test_size = int(len(data_set) * test_percent)

    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size).take(val_size)
    test_ds = data_set.skip(train_size + val_size).take(test_size)

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
        ds_saturation = (
            ds.map(random_saturation, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
        ds = ds.concatenate(ds_saturation)

    if brightness:
        ds_brightness = (
            ds.map(random_brightness, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
        ds = ds.concatenate(ds_brightness)

    if rgb:
        ds_gray = (
            ds.map(rgb_to_gray_scale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
        ds = ds.concatenate(ds_gray)

    if rot:
        ds_rot = (
            ds.map(image_rot_on_dataset, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
        ds = ds.concatenate(ds_rot)

    if hue:
        ds_hue = (
            ds.map(random_hue, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
        ds = ds.concatenate(ds_hue)

    return ds.prefetch(buffer_size=AUTOTUNE)


def prepare_layers(ds, rot=False, bright=False, flip=False):
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


def train_and_save_model(img_height, img_width, num_classes, epochs, train_ds, val_ds, optimizer):
    model = create_HyperModel(img_height, img_width, num_classes)  # tu nalezy wymienic model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("config_results")
    model.summary()
    return model, history


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
    augmented_ds = prepare_layers(data, rot=True, bright=True, flip=True)

    train_ds, val_ds, test_ds = split_data(augmented_ds, 0.7, 0.2, 0.1)

    class_names = data.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    num_classes = len(class_names)

    # model creation

    img_height = 180
    img_width = 180
    num_classes = 5
    epochs = 10

    model, history = train_and_save_model(img_height, img_width, num_classes, epochs, train_ds, val_ds,
                                          optimizer='adam')
    tuner = kt.Hyperband(create_HyperModel,
                         objective='val_accuracy',
                         max_epochs=epochs,
                         factor=3,
                         directory='auto_trening',
                         project_name='aiir')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, epochs=50, validation_data=val_ds)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train_ds, epochs=best_epoch, validation_data=val_ds)

    eval_result = hypermodel.evaluate(val_ds)
    print("[test loss, test accuracy]:", eval_result)

    with open(f'models/hyper_model - {epochs}.txt', 'a') as f:
        json.dump({'params': model.get_config(), 'history': history.history}, f)

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
