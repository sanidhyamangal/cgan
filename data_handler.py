import tensorflow as tf # for deep learning 
import pandas as pd # for dataframes based ops


def process_images_labels(images:tf.Tensor, labels:tf.Tensor):
    images = tf.reshape((images / 127.5) - 1, [28, 28, 1])
    return images, labels

def data_loader_csv(df_path: str, batch_size=64, shuffle=True):

    data = pd.read_csv(df_path)

    data_set_labels, data_set_data = tf.convert_to_tensor(
        data.iloc[:, 0],
        tf.int32), tf.convert_to_tensor(data.iloc[:, 1:], tf.float32)

    # free up the memory for train set test_set
    del data

    # create a train and test dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (data_set_data, data_set_labels)).map(process_images_labels).shuffle(60000).batch(batch_size)

    del data_set_data
    del data_set_labels

    # return dataset
    return dataset