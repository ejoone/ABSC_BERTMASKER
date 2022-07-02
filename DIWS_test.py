# DIWS model for domain accuracy test
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


def main():
    apex_domain = ["Apex", 2004, 0.0005, 0.9, 10, 30]
    camera_domain = ["Camera", 2004, 0.001, 0.9, 10, 15]
    hotel_domain = ["hotel", 2015, 0.001, 0.85, 15, 5]
    nokia_domain = ["Nokia", 2004, 0.001, 0.95, 20, 10]
    domains = [apex_domain, camera_domain, hotel_domain, nokia_domain]

    results = []

    for domain in domains:
        source_domain = "Creative"
        source_year = 2004
        target_domain = domain[0]
        target_year = domain[1]
        learning_rate_hyper = domain[2]
        momentum_hyper = domain[3]
        epochs_hyper = domain[4]
        batch_size_hyper = domain[5]

        # input data should be an integreated word embedding data with source and target domains
        source_path_train = "data/programGeneratedData/BERT/" + source_domain + "/768_" + source_domain + "_train_" + str(
            source_year) + "_BERT.txt"
        target_path_train = "data/programGeneratedData/BERT/" + target_domain + "/768_" + target_domain + "_train_" + str(
            target_year) + "_BERT.txt"
        source_path_test = "data/programGeneratedData/BERT/" + source_domain + "/768_" + source_domain + "_test_" + str(
            source_year) + "_BERT.txt"
        target_path_test = "data/programGeneratedData/BERT/" + target_domain + "/768_" + target_domain + "_test_" + str(
            target_year) + "_BERT.txt"
        FLAGS.source_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + source_domain + "_" + str(
            source_year) + "_" + str(FLAGS.embedding_dim) + ".txt"
        FLAGS.target_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + target_domain + "_" + str(
            target_year) + "_" + str(FLAGS.embedding_dim) + ".txt"

        tf.keras.backend.clear_session()
        model = build_model()

        inputs_x, inputs_y, sen_len, source_sen_len, target_sen_len, numdata_source, numdata_target = load_data(
            source_path_train, target_path_train)
        test_x, test_y, _, _, _, _, _ = load_data(source_path_test, target_path_test)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate_hyper, beta_1=momentum_hyper),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(inputs_x, inputs_y, epochs=epochs_hyper, batch_size=batch_size_hyper)

        score = model.evaluate(test_x, test_y, batch_size=batch_size_hyper)
        # print('Test loss:', score[0])
        print('Test accuracy for', domain[0], ' is: ', score[1])
        results.append(score[1])

    print(results)


if __name__ == "__main__":
    main()
