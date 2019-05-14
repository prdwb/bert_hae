from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

def bert_rep(bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        history_answer_marker=history_answer_marker,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()   
    return final_hidden

def bert_segment_rep(final_hidden):
    first_token_tensor = tf.squeeze(final_hidden[:, 0:1, :], axis=1)    
    return first_token_tensor

def cqa_model(final_hidden):
    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/cqa/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/cqa/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


# def cqa_model(bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings):
#     final_hidden = bert_rep(bert_config, is_training, input_ids, input_mask, segment_ids, history_answer_marker, use_one_hot_embeddings)

#     final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
#     batch_size = final_hidden_shape[0]
#     seq_length = final_hidden_shape[1]
#     hidden_size = final_hidden_shape[2]

#     output_weights = tf.get_variable(
#         "cls/cqa/output_weights", [2, hidden_size],
#         initializer=tf.truncated_normal_initializer(stddev=0.02))

#     output_bias = tf.get_variable(
#         "cls/cqa/output_bias", [2], initializer=tf.zeros_initializer())

#     final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
#     logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
#     logits = tf.nn.bias_add(logits, output_bias)

#     logits = tf.reshape(logits, [batch_size, seq_length, 2])
#     logits = tf.transpose(logits, [2, 0, 1])

#     unstacked_logits = tf.unstack(logits, axis=0)

#     (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

#     return (start_logits, end_logits)
