import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from datasets import load_pamap2, create_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def representative_dataset():
    x_train, y_train, x_test, y_test = load_pamap2(window=config['window'], overlap=config['window'] - 1, train_subjects=['subject101'], test_subjects=['subject102'])

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 3)
    x_train = np.array_split(x_train, config['batch_size'])
    for i in x_train:
        yield [i.astype(np.float32)]


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def run():
    f1s = []
    groups = np.array(['subject101', 'subject102', 'subject103', 'subject104', 'subject105', 'subject106', 'subject107',
                       'subject108', ])
    for i, sid in reversed(list(enumerate(groups))):

        n_channels = 3

        input1 = Input(shape=(config['window'], 1, n_channels))
        enc1 = Conv2D(activity_regularizer=l2(config['act_reg']), kernel_regularizer=l2(config['kern_reg']), filters=32,
                      kernel_size=config['kernel_size'], activation='relu')(input1)
        enc1 = MaxPooling2D(pool_size=(1, 1))(enc1)
        enc1 = Dropout(config['dropout'])(enc1)
        enc1 = Flatten()(enc1)
        enc1 = Dense(config['clf_1'], activity_regularizer=l2(config['act_reg']), activation='relu')(enc1)
        enc1 = Dropout(config['dropout'])(enc1)
        enc1 = Dense(12, activation='softmax', name='output')(enc1)
        model = Model(inputs=input1, outputs=enc1)
        optimizer = Adam(lr=config['learning_rate'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', get_f1])

        test_subject = sid
        train_subjects = []
        for j, isid in enumerate(groups):
            if i != j:
                train_subjects.append(isid)

        print(train_subjects)
        print(test_subject)

        x_train, y_train, x_test, y_test = load_pamap2(window=config['window'], overlap=config['window'] - 1,
                                                       train_subjects=train_subjects,
                                                       test_subjects=[test_subject])
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, n_channels)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, n_channels)
        model.fit(x_train, y_train, verbose=1, epochs=config['epochs'],
                  batch_size=config['batch_size'])
        y_pred = model.predict(x_test)
        f1 = get_f1(y_test, y_pred)
        print(f1)
        f1s.append(f1)
        break

    print("The F1 scores from each split", f1s)
    print("The mean F1 score", np.mean(f1s))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    with open('cnn_quant_dynamic.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_int_model = converter.convert()
    with open('cnn_quant_int.tflite', 'wb') as f:
        f.write(tflite_quant_int_model)


config = {
    'raw_data_path': '../PAMAP2_Dataset/Protocol',
    'window': 256,
    'overlap': 255,
    'test_overlap': 0,
    'epochs': 100,
    'batch_size': 4096,
    'learning_rate': 1e-4,
    'dropout': 0.5,
    'act_reg': 0.01,
    'clf_1': 16,
    'kernel_size': (64, 1),
    'kern_reg': 0.05,

}



if not os.path.exists('processed'):
    create_dataset(config['raw_data_path'], config['window'], config['overlap'], config['test_overlap'])

run()
