import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Flatten
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.metrics import Precision, Recall

from recognition.constants import TRAIN_CHECKPOINTS_PATH


class L1DistanceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def build_image_embedding_model():
    input_layer = Input(shape=(100, 100, 3), name='Input Image')

    conv_layer_1 = Conv2D(64, (10, 10), activation='relu')(input_layer)
    pool_layer_1 = MaxPooling2D(64, (2, 2), padding='same')(conv_layer_1)
    conv_layer_2 = Conv2D(128, (7, 7), activation='relu')(pool_layer_1)
    pool_layer_2 = MaxPooling2D(64, (2, 2), padding='same')(conv_layer_2)
    conv_layer_3 = Conv2D(128, (4, 4), activation='relu')(pool_layer_2)
    pool_layer_3 = MaxPooling2D(64, (2, 2), padding='same')(conv_layer_3)
    conv_layer_4 = Conv2D(256, (4, 4), activation='relu')(pool_layer_3)

    flatten_layer = Flatten()(conv_layer_4)
    dense_layer = Dense(4096, activation='sigmoid')(flatten_layer)

    return Model(
        inputs=[input_layer],
        outputs=[dense_layer],
        name='image_embedding'
    )


def build_siamese_network_model():
    input_image = Input(name='Input Image', shape=(100, 100, 3))
    validation_image = Input(name='Validation Image', shape=(100, 100, 3))

    embedding_model = build_image_embedding_model()
    input_embedding = embedding_model(input_image)
    validation_embedding = embedding_model(validation_image)

    siamese_layer = L1DistanceLayer()
    siamese_layer._name = 'l1_distance'
    embedding_distances = siamese_layer(input_embedding, validation_embedding)

    classifier = Dense(1, activation='sigmoid')(embedding_distances)

    return Model(
        inputs=[input_image, validation_image],
        outputs=classifier,
        name='SiameseNetwork',
    )


@tf.function
def run_train_step(model, batch, loss_function, optimizer):
    with tf.GradientTape() as tape:
        sample_data, label = batch[0:2], batch[2]
        prediction = model(sample_data, training=True)
        loss = loss_function(label, prediction)

    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    return loss


def evaluate_siamese_network(model, test_data):
    precision_metric = Precision()
    recall_metric = Recall()

    for input, validation, label in test_data.as_numpy_iterator():
        prediction = model.predict([input, validation])
        precision_metric.update_state(label, prediction)
        recall_metric.update_state(label, prediction)

    print(
        f'''\nThe Siamese Neural Network was trained!
    - Precision: {round(precision_metric.result().numpy() * 100, 2)}%
    - Recall: {round(recall_metric.result().numpy() * 100, 2)}%'''
    )


def run_siamese_network_training(model, train_data, test_data, n_epochs=50):
    binary_cross_entropy = tf.losses.BinaryCrossentropy()
    adam_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(opt=adam_optimizer, siamese_model=model)
    checkpoint_prefix = os.path.join(TRAIN_CHECKPOINTS_PATH, 'ckpt')

    for epoch in range(1, n_epochs + 1):
        print(f'\nEPOCH {epoch}/{n_epochs}: Training...')
        progbar = tf.keras.utils.Progbar(len(train_data))

        for i, batch in enumerate(train_data):
            run_train_step(
                model=model,
                batch=batch,
                loss_function=binary_cross_entropy,
                optimizer=adam_optimizer,
            )
            progbar.update(i + 1)

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    evaluate_siamese_network(model=model, test_data=test_data)
    model.save('siamese_network.h5')
