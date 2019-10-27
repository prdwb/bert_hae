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
from copy import deepcopy

from cqa_flags import FLAGS
from cqa_supports import *
from cqa_model import *
from cqa_gen_batches import *

class CQAExample(object):
    """A single training/test example."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 history_answer_marker=None,
                 metadata=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.history_answer_marker = history_answer_marker
        self.metadata = metadata

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.history_answer_marker:
            s += ', history_answer_marker: {}'.format(json.dumps(self.history_answer_marker))
        if self.metadata:
            s += ', metadata: ' + json.dumps(self.metadata)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 history_answer_marker=None,
                 metadata=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.history_answer_marker = history_answer_marker
        self.metadata = metadata
        
def read_quac_examples(input_file, is_training):
    """Read a QuAC json file into a list of CQAExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    if FLAGS.load_small_portion:
        input_data = input_data[:10]
        # print('input_data:', input_data)
        tf.logging.warning('<<<<<<<<<< load_small_portion is on! >>>>>>>>>>')
    for entry in input_data:
        # An additional "CANNOTANSWER" has been added in QuAC data, so no need to append one.
        entry = entry['paragraphs'][0]
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
            
        ############################################################
        # convert the convasational QAs to squad format, with history
        ############################################################

        questions = [(item['question'], item['id']) for item in entry['qas']] # [(question, question_id), ()]
        answers = [(item['orig_answer']['text'], item['orig_answer']['answer_start']) for item in entry['qas']]
        followups = [item['followup'] for item in entry['qas']]
        yesnos = [item['yesno'] for item in entry['qas']]

        qas = []
        for i, (question, answer, followup, yesno) in enumerate(zip(questions, answers, followups, yesnos)):
            metadata = {'turn': i + 1, 'history_turns': [], 'tok_history_answer_markers':[], 
                        'followup': followup, 'yesno': yesno, 'history_turns_text': []}
            # if FLAGS.use_RL:
            #     start_index = 0
            # else:
            #     start_index = 0 if i - int(FLAGS.history) < 0 else i - int(FLAGS.history)
            
            end_index = i
            question_with_histories = ''
            
            history_answer_marker = None

            start_index = 0 # we read all the histories no matter we use RL or not. we will make approporiate selections afterwards
            history_answer_marker = []
            for history_turn, (each_answer, each_question) in enumerate(
                zip(answers[start_index: end_index], questions[start_index: end_index])):

                # [history_answer_start, history_answer_end, history_answer_text]
                each_marker = [each_answer[1], each_answer[1] + len(each_answer[0]), each_answer[0]]
                history_answer_marker.append(each_marker)
                metadata['history_turns'].append(history_turn + start_index + 1)
                metadata['history_turns_text'].append((each_question[0], each_answer[0])) #[(q1, a1), (q2, a2), ...]
            
            # add the current question
            question_with_histories += question[0]
            qas.append({'id': question[1], 'question': question_with_histories, 'answers': [{'answer_start': answer[1], 'text': answer[0]}],
                        'history_answer_marker': history_answer_marker, 'metadata': metadata})

        for qa in qas:
            qas_id = qa["id"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            
            # if is_training:
            # we read in the groundtruth answer bothing druing training and predicting, because we need to compute acc and f1 at predicting time.
            if len(qa["answers"]) != 1:
                raise ValueError(
                    "For training, each question should have exactly 1 answer.")
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            
            if is_training and actual_text.find(cleaned_answer_text) == -1:
                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                continue
                
            # we construct a tok_history_answer_marker to store the aggregated history answer markers for a question.
            # we also construct each_tok_history_answer_marker to store a single history answer marker.
            tok_history_answer_marker = [0] * len(doc_tokens)

            for marker_index, marker in enumerate(qa['history_answer_marker']):
                each_tok_history_answer_marker = [0] * len(doc_tokens)
                history_orig_answer_text = marker[2]
                history_answer_offset = marker[0]
                history_answer_length = len(history_orig_answer_text)
                history_start_position = char_to_word_offset[history_answer_offset]
                history_end_position = char_to_word_offset[history_answer_offset + history_answer_length - 1]
                history_actual_text = " ".join(doc_tokens[history_start_position:(history_end_position + 1)])
                history_cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(history_orig_answer_text))
                if history_actual_text.find(history_cleaned_answer_text) != -1:
                    tok_history_answer_marker = tok_history_answer_marker[: history_start_position] + \
                                        [1] * (history_end_position - history_start_position + 1) + \
                                        tok_history_answer_marker[history_end_position + 1 :]
                    each_tok_history_answer_marker = each_tok_history_answer_marker[: history_start_position] + \
                                        [1] * (history_end_position - history_start_position + 1) + \
                                        each_tok_history_answer_marker[history_end_position + 1 :]
                    assert len(tok_history_answer_marker) == len(doc_tokens)
                    assert len(each_tok_history_answer_marker) == len(doc_tokens)
                    qa['metadata']['tok_history_answer_markers'].append(each_tok_history_answer_marker)
                else:
                    tf.logging.warning("Could not find history answer: '%s' vs. '%s'", history_actual_text, history_cleaned_answer_text)                                    

            example = CQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                history_answer_marker=tok_history_answer_marker,
                metadata=qa['metadata'])
            examples.append(example)
            # print(example)
    return examples
    

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        metadata = example.metadata
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        
        history_answer_marker = example.history_answer_marker
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_history_answer_marker = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                all_history_answer_marker.append(history_answer_marker[i])

        tok_start_position = None
        tok_end_position = None
        
        # if is_training:
        # we do this for both training and predicting, because we need also start/end position at testing time to compute acc and f1
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
            
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            marker = []
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            marker.append(0)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                marker.append(0)
                segment_ids.append(0)
            tokens.append("[SEP]")
            marker.append(0)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                marker.append(all_history_answer_marker[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            marker.append(0)
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                marker.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(marker) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                    example.end_position < doc_start or
                    example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
            else:
                # when predicting, we donot throw out any doc span to prevent label leaking
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

#             if example_index < 20:
#                 tf.logging.info("*** Example ***")
#                 tf.logging.info("unique_id: %s" % (unique_id))
#                 tf.logging.info("example_index: %s" % (example_index))
#                 tf.logging.info("doc_span_index: %s" % (doc_span_index))
#                 tf.logging.info("tokens: %s" % " ".join(
#                     [tokenization.printable_text(x) for x in tokens]))
#                 tf.logging.info("token_to_orig_map: %s" % " ".join(
#                     ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
#                 tf.logging.info("token_is_max_context: %s" % " ".join([
#                     "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
#                 ]))
#                 tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#                 tf.logging.info(
#                     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
#                 tf.logging.info(
#                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#                 tf.logging.info(
#                     "marker: %s" % " ".join([str(x) for x in marker]))
#                 tokens_where_marker_is_true = [token for (token, token_marker) in zip(tokens, marker) if token_marker == 1]
#                 tf.logging.info("tokens_where_marker_is_true: %s" % " ".join(
#                     [tokenization.printable_text(x) for x in tokens_where_marker_is_true]))
#                 if is_training:
#                     answer_text = " ".join(tokens[start_position:(end_position + 1)])
#                     tf.logging.info("start_position: %d" % (start_position))
#                     tf.logging.info("end_position: %d" % (end_position))
#                     tf.logging.info(
#                         "answer: %s" % (tokenization.printable_text(answer_text)))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    history_answer_marker=marker,
                    metadata=metadata))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, for_reward=False):
    """Write final predictions to the json file."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        #if len(features) == 0:
        #    continue

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            if for_reward:
                continue
            if FLAGS.dataset.lower() == 'coqa':
                nbest.append(_NbestPrediction(text="unknown", start_logit=0.0, end_logit=0.0))
            elif FLAGS.dataset.lower() == 'quac':
                nbest.append(_NbestPrediction(text="invalid", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        # convert to official evaluation format
        if FLAGS.dataset.lower() == 'coqa':
            converted = []
            for key, value in all_predictions.items():
                converted.append({'id': key[:30], 'turn_id': int(key[30:]), 'answer': value})
                writer.write(json.dumps(converted, indent=4) + "\n")
        elif FLAGS.dataset.lower() == 'quac':
            converted = {}
            for key, value in all_predictions.items():
                dialog_id = key[:36]
                if dialog_id not in converted:
                    converted[dialog_id] = {'best_span_str': [], 'qid': [], 'followup': [], 'yesno': []}
                converted[dialog_id]['best_span_str'].append(value)
                converted[dialog_id]['qid'].append(key)
                converted[dialog_id]['followup'].append('x')
                converted[dialog_id]['yesno'].append('y')
            for key, value in converted.items():
                writer.write(json.dumps(value) + '\n')
            
        

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def convert_features_to_feed_dict(features):
    batch_unique_ids, batch_input_ids, batch_input_mask = [], [], []
    batch_segment_ids, batch_start_positions, batch_end_positions, batch_history_answer_marker = [], [], [], []

    for feature in features:
        batch_unique_ids.append(feature.unique_id)
        batch_input_ids.append(feature.input_ids)
        batch_input_mask.append(feature.input_mask)
        batch_segment_ids.append(feature.segment_ids)
        batch_start_positions.append(feature.start_position)
        batch_end_positions.append(feature.end_position)
        batch_history_answer_marker.append(feature.history_answer_marker)
    
    feed_dict = {'unique_ids': batch_unique_ids, 'input_ids': batch_input_ids, 
              'input_mask': batch_input_mask, 'segment_ids': batch_segment_ids, 
              'start_positions': batch_start_positions, 'end_positions': batch_end_positions, 
              'history_answer_marker': batch_history_answer_marker}
    return feed_dict

def convert_examples_to_example_variations(examples, max_considered_history_turns):
    # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
    # an example variation is "question + passage + markers (M3)"
    # meaning that we only have one marker for each example variation
    # because we want to make a binary choice for every example variation,
    # and combine all variations to form an example
    
    new_examples = []
    for example in examples:
        # if the example is the first question in the dialog, it does not contain history answers, 
        # so we simply append it.
        if len(example.metadata['tok_history_answer_markers']) == 0:
            example.metadata['history_turns'] = []
            new_examples.append(example)
        else:
            for history_turn, marker, history_turn_text in zip(
                    example.metadata['history_turns'][- max_considered_history_turns:], 
                    example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                    example.metadata['history_turns_text'][- max_considered_history_turns:]):
                each_new_example = deepcopy(example)
                each_new_example.history_answer_marker = marker
                each_new_example.metadata['history_turns'] = [history_turn]
                each_new_example.metadata['tok_history_answer_markers'] = [marker]
                each_new_example.metadata['history_turns_text'] = [history_turn_text]
                new_examples.append(each_new_example)
    return new_examples

def convert_examples_to_variations_and_then_features(examples, tokenizer, max_seq_length, 
                                doc_stride, max_query_length, max_considered_history_turns, is_training):
    # different from the "convert_examples_to_features" in cqa_supports.py, we return two masks with the feature (example/variaton trackers).
    # the first mask is the example index, and the second mask is the variation index. Wo do this to keep track of the features generated
    # by different examples and variations.
    
    all_features = []
    example_features_nums = [] # keep track of how many features are generated from the same example (regardless of example variations)
    example_tracker = []
    variation_tracker = []
    # matching_signals_dict = {}
    unique_id = 1000000000
    
    
    # when training, we shuffle the data for more stable training.
    # we shuffle here so that we do not need to shuffle when generating batches
    num_examples = len(examples)    
    if is_training:
        np.random.seed(0)
        idx = np.random.permutation(num_examples)
        examples_shuffled = np.asarray(examples)[idx]
    else:
        examples_shuffled = np.asarray(examples)
    
    for example_index, example in enumerate(examples_shuffled):
        example_features_num = []
        variations = convert_examples_to_example_variations([example], max_considered_history_turns)
        for variation_index, variation in enumerate(variations):
            features = convert_examples_to_features([variation], tokenizer, max_seq_length, doc_stride, max_query_length, is_training)
            # matching_signals = extract_matching_signals(variation, glove, tfidf_vectorizer)
            # matching_signals_dict[(example_index, variation_index)] = matching_signals
            
            # the example_index and unique_id in features are wrong due to the generation of example variations.
            # we fix them here.
            for i in range(len(features)):
                features[i].example_index = example_index
                features[i].unique_id = unique_id
                unique_id += 1
            all_features.extend(features)
            variation_tracker.extend([variation_index] * len(features))
            example_tracker.extend([example_index] * len(features))
            example_features_num.append(len(features))
        # every variation of the same example should generate the same amount of features
        assert len(set(example_features_num)) == 1
        example_features_nums.append(example_features_num[0]) 
    assert len(all_features) == len(example_tracker)
    assert len(all_features) == len(variation_tracker)
    return all_features, example_tracker, variation_tracker, example_features_nums





def get_selected_example_features_without_actions(batch_features, batch_example_tracker, batch_variation_tracker):
    
    prev_e_tracker, prev_v_tracker = None, None
    relative_selected_pos = [] # the relative position of the selected history compared to the current turn
    handled_variations = {} # each example variation could have more than one features, but we only store the "relative_selected_pos" once 
    f_tracker = 0 # feature tracker, denotes the feature index for each variation
    
    selected_example_features_dict = {}
    for feature, e_tracker, v_tracker in zip(batch_features, batch_example_tracker, batch_variation_tracker):
        # get the f_tracker
        if e_tracker == prev_e_tracker and v_tracker == prev_v_tracker:
            f_tracker += 1
        else:
            f_tracker = 0
        prev_e_tracker, prev_v_tracker = e_tracker, v_tracker

        # we append a feature that is without any history, so we can build upon this if other histories are chosen
        if e_tracker not in selected_example_features_dict:                    
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker] = [feature_without_marker]
        elif f_tracker >= len(selected_example_features_dict[e_tracker]):
            feature_without_marker = deepcopy(feature)
            feature_without_marker.history_answer_marker = np.zeros(len(feature.history_answer_marker))
            feature_without_marker.metadata['history_turns'] = []
            selected_example_features_dict[e_tracker].append(feature_without_marker)
        
        # add history answer markers, a turn is selected if it either in immediate previous j turns or have large tfidf score
        if len(feature.metadata['history_turns']) > 0 and \
             (feature.metadata['history_turns'][0] >= feature.metadata['turn'] - FLAGS.history):
                
            curr_feature = deepcopy(selected_example_features_dict[e_tracker][f_tracker])
            

            curr_feature.history_answer_marker = curr_feature.history_answer_marker + \
                                                 np.asarray(feature.history_answer_marker)
            # when the markers have overlaps, the corresponding bit would be 2 after added together. we need to set it to one.
            curr_feature.history_answer_marker = curr_feature.history_answer_marker != 0
            curr_feature.history_answer_marker = curr_feature.history_answer_marker.astype(int)

            curr_feature.metadata['history_turns'].append(feature.metadata['history_turns'][0])
            selected_example_features_dict[e_tracker][f_tracker] = curr_feature
            
            if (e_tracker, v_tracker) not in handled_variations:
                # we always choose the current turn
                relative_selected_pos.append(0)
                
                handled_variations[(e_tracker, v_tracker)] = 1
                if len(feature.metadata['history_turns']) > 0:
                    relative_selected_pos.append(feature.metadata['turn'] - feature.metadata['history_turns'][0])

    selected_example_features = []
    for features in selected_example_features_dict.values():
        selected_example_features.extend(features)
        
    return selected_example_features, relative_selected_pos

