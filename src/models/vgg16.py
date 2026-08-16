def build_vgg16_classifier(num_classes: int = 3, image_size: int = 128):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("Install tensorflow to train the VGG16 baseline.") from exc
    base = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3))
    base.trainable = False
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="vgg16_soil_moisture")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model