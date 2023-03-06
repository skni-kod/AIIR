from keras import layers
from keras import Sequential


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
