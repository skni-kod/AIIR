#%%
import os
import tensorflow as tf
from model import *
import json
#%%
def split_data(data_set, train_percent, val_percent, test_percent):
    data_set = data_set.shuffle(buffer_size=len(data_set), reshuffle_each_iteration=False)

    train_size = int(len(data_set) * train_percent)
    val_size = int(len(data_set) * val_percent)
    test_size = int(len(data_set) * test_percent)

    train_ds = data_set.take(train_size)
    val_ds = data_set.skip(train_size).take(val_size)
    test_ds = data_set.skip(train_size + val_size).take(test_size)

    return train_ds, val_ds, test_ds


def prepare_using_layers(ds, directory_name="augmented_dataset", rot=False, bright=False, flip=False):
    """
    Function augments given dataset using Sequential model layers
    :param directory_name: name of a directory where augmented images will be saved
    :param ds: dataset to be augmented
    :param rot: determines whether to apply random rotation
    :param bright: determines whether to apply random brightness
    :param flip: determines whether to apply random flips
    :return: augmented dataset
    """
    batch, _ = next(iter(ds))
    batch_size = batch.shape[0]
    img_height = batch.shape[1]
    img_width = batch.shape[2]

    if not os.path.exists('datasets/' + directory_name):
        os.makedirs(directory_name)
    labels = ds.class_names
    for label in labels:
        if not os.path.exists(f'datasets/{directory_name}/{label}'):
            os.makedirs(f'datasets/{directory_name}/{label}')
    counter = 0
    for image_batch, labels_batch in ds:
        for image, label in zip(image_batch, labels_batch):
            tf.keras.preprocessing.image.save_img(f'datasets/{directory_name}/{labels[label]}/{counter}.png', image)
            counter += 1
    if rot:
        random_rot = tf.keras.Sequential([
            layers.RandomRotation(0.2),
        ])
        augmented_ds = ds.map(lambda x, y: (random_rot(x), y), num_parallel_calls=AUTOTUNE)
        for image_batch, labels_batch in augmented_ds:
            for image, label in zip(image_batch, labels_batch):
                tf.keras.preprocessing.image.save_img(f'datasets/{directory_name}/{labels[label]}/{counter}.png', image)
                counter += 1
    if bright:
        random_bright = tf.keras.layers.RandomBrightness(factor=0.2)
        augmented_ds = ds.map(lambda x, y: (random_bright(x), y), num_parallel_calls=AUTOTUNE)
        for image_batch, labels_batch in augmented_ds:
            for image, label in zip(image_batch, labels_batch):
                tf.keras.preprocessing.image.save_img(f'datasets/{directory_name}/{labels[label]}/{counter}.png', image)
                counter += 1
    if flip:
        random_flip = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
        ])
        augmented_ds = ds.map(lambda x, y: (random_flip(x), y), num_parallel_calls=AUTOTUNE)
        for image_batch, labels_batch in augmented_ds:
            for image, label in zip(image_batch, labels_batch):
                tf.keras.preprocessing.image.save_img(f'datasets/{directory_name}/{labels[label]}/{counter}.png', image)
                counter += 1

    data_dir = 'datasets/' + directory_name
    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        batch_size=batch_size,
        image_size=(img_height, img_width))

    return data
#%%
path = "datasets/improved_gestures_dataset"  # Remember to recreate or change the path to the actual dataset
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
#%%
# augmented_ds = prepare(data, shuffle=False, rot=True , rgb=True, brightness=False, saturation=True, hue=False)
augmented_ds = prepare_using_layers(data, directory_name="gestures_dataset", rot=True, bright=True, flip=True)
#%%
counter = 0
for image_batch, labels_batch in augmented_ds:
    for image, label in zip(image_batch, labels_batch):
        counter += 1
print(counter)
#%%
train_ds, val_ds, test_ds = split_data(augmented_ds, 0.70, 0.2, 0.1)

class_names = data.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1. / 255)

num_classes = len(class_names)
#%%
epochs = 5

functions = [create_model1, create_model2, create_model3, create_model4, create_model5]
filters = [8, 16, 32]
pool_size = [2, 4, 8]
dense_units = [32, 64, 128, 256]
counter = 1
total = len(filters) * len(pool_size) * len(dense_units) * len(functions)
eval_results = dict()
#%%
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

                evaluation = model.evaluate(test_ds)
                eval_results[f'model_{counter}'] = {"train_accuracy": history.history['accuracy'][-1],
                                                     "val_accuracy": history.history['val_accuracy'][-1],
                                                     "test_loss": evaluation[0], "test_accuracy": evaluation[1]}

                if not os.path.exists('models/'):
                    os.makedirs('models')
                model.save('models/model_' + str(counter))
                print(f"Progress: {counter}/{total}")
                counter += 1

with open('models/results.json', 'w') as out_file:
    json.dump(eval_results, out_file)
