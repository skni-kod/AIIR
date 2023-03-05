from keras import Sequential
import tensorflow as tf
from keras import layers
import pathlib
def create_model1(img_height, img_width, num_classes, filters=16, pool_size=2, dense_units=128):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(2 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(4 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model

def create_model2(img_height, img_width, num_classes, filters=32, pool_size=2, dense_units=64):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(2 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(4 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model
def create_model3(img_height, img_width, num_classes, filters=32, pool_size=2, dense_units=256):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(2 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(4 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model

def create_model4(img_height, img_width, num_classes, filters=16, pool_size=2, dense_units=128):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(2 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model
def create_model5(img_height, img_width, num_classes, filters=16, pool_size=2, dense_units=128):
    model = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(2 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(4 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(8 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(16 * filters, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model


if __name__ == '__main__':
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    epochs = 10
    num_classes = 5

    functions = [create_model1, create_model2, create_model3, create_model4, create_model5]
    filters = [8, 16, 32]
    pool_size = [2, 4, 8]
    dense_units = [32, 64, 128, 256]
    counter = 1
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

                    model.save('model ' + str(counter)+'.h5')
                    counter += 1
