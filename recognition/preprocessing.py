import tensorflow as tf
from tensorflow.data import Dataset

from recognition.constants import ANCHOR_PATH, POSITIVE_PATH, NEGATIVE_PATH


def get_data_samples():
    anchor_samples = Dataset.list_files(ANCHOR_PATH + '/*.jpg').take(300)
    positive_samples = Dataset.list_files(POSITIVE_PATH + '/*.jpg').take(300)
    negative_samples = Dataset.list_files(NEGATIVE_PATH + '/*.jpg').take(300)

    return (anchor_samples, positive_samples, negative_samples)


def preprocess_image(path):
    image_file = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_file)

    image = tf.image.resize(image, (100, 100))
    image = image / 255.0

    return image


def preprocess_data_sample(image1, image2, label):
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    return (image1, image2, label)


def create_labeled_data():
    anchor_samples, positive_samples, negative_samples = get_data_samples()

    samples_size = len(anchor_samples)
    true_labels = Dataset.from_tensor_slices(tf.ones(samples_size))
    false_labels = Dataset.from_tensor_slices(tf.zeros(samples_size))

    positives = Dataset.zip((anchor_samples, positive_samples, true_labels))
    negatives = Dataset.zip((anchor_samples, negative_samples, false_labels))

    data = positives.concatenate(negatives).map(preprocess_data_sample)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    return data


def split_train_test(data, train_size=0.7):
    train_data = data.take(round(len(data) * train_size))
    test_data = data.skip(round(len(data) * train_size))
    test_data = test_data.take(round(len(data) * (1 - train_size)))

    train_data = train_data.batch(16)
    test_data = test_data.batch(16)

    train_data = train_data.prefetch(8)
    test_data = test_data.prefetch(8)

    return train_data, test_data


def create_dataset():
    data = create_labeled_data()
    train_data, test_data = split_train_test(data)

    return train_data, test_data
