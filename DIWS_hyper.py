# DIWS model for hyperparameter tuning
#
# https://github.com/ejoone/DIWS-ABSC

import tensorflow as tf
import pandas as pd
import numpy as np
from config import *
from utils import load_inputs_bertmasker, load_w2v, load_inputs_attentionmasker

reduce_size = 30


def build_model():
    # Input Layer
    input_layer = tf.keras.layers.Input(shape=(FLAGS.max_sentence_len, FLAGS.embedding_dim))
    reduce_input = tf.keras.layers.Dense(reduce_size, activation='relu', trainable=True)(input_layer)
    flat_layer = tf.keras.layers.Flatten()(reduce_input)
    # Attention Layer
    attention_probs = tf.keras.layers.Dense(FLAGS.max_sentence_len * reduce_size, activation='softmax',
                                            name='attention_vec')(flat_layer)
    attention_mul = tf.keras.layers.multiply([flat_layer, attention_probs])
    # Full Connected Layer
    fc_attention_mul = tf.keras.layers.Dense(64)(attention_mul)
    y = tf.keras.layers.Dense(1, activation='sigmoid')(fc_attention_mul)
    return tf.keras.Model(inputs=[input_layer], outputs=y)


def load_data(source_path, target_path):
    source_word_id_mapping, source_w2v = load_w2v(FLAGS.source_embedding, FLAGS.embedding_dim)
    target_word_id_mapping, target_w2v = load_w2v(FLAGS.target_embedding, FLAGS.embedding_dim)
    source_word_embedding = tf.constant(source_w2v, name='source_word_embedding')
    target_word_embedding = tf.constant(target_w2v, name='target_word_embedding')
    word_embedding = tf.concat([source_word_embedding, target_word_embedding], axis=0)

    source_x, source_sen_len, source_y = load_inputs_attentionmasker(source_path, source_word_id_mapping,
                                                                     FLAGS.max_sentence_len, 'TC', domain='source')
    target_x, target_sen_len, target_y = load_inputs_attentionmasker(target_path, target_word_id_mapping,
                                                                     FLAGS.max_sentence_len, 'TC', domain='target')

    num_word_emb_source, _ = source_word_embedding.get_shape()
    target_x_v2 = target_x + num_word_emb_source
    x_concat = np.concatenate((source_x, target_x_v2), axis=0)
    sen_len_concat = np.concatenate((source_sen_len, target_sen_len), axis=0)
    y_concat = np.concatenate((source_y, target_y), axis=0)

    x_concat_tensor = tf.convert_to_tensor(x_concat, dtype=tf.int32)

    inputs = tf.nn.embedding_lookup(word_embedding, x_concat_tensor)
    inputs_nparray = inputs.eval(session=tf.Session())

    return inputs_nparray, y_concat, sen_len_concat, source_sen_len, target_sen_len, len(source_y), len(target_y)


def main(source_path_train, target_path_train, source_path_test, target_path_test, learning_rate, momentum,
         epochs_hyper, batch_size_hyper):
    tf.keras.backend.clear_session()
    model = build_model()
    # model.summary()

    inputs_x, inputs_y, sen_len, source_sen_len, target_sen_len, numdata_source, numdata_target = load_data(
        source_path_train, target_path_train)
    test_x, test_y, _, _, _, _, _ = load_data(source_path_test, target_path_test)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=momentum), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(inputs_x, inputs_y, epochs=epochs_hyper, batch_size=batch_size_hyper)

    score = model.evaluate(test_x, test_y, batch_size=batch_size_hyper)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score[1]

# model.fit(inputs_x, inputs_y, epochs=10, batch_size=10, validation_split=0.5, verbose=2)
