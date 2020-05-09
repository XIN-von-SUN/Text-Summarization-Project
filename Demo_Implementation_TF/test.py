import os
print(os.getcwd())
import time
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import tensorflow_hub as hub
from collections import Counter
import nltk
nltk.download('punkt')

import data_process
import model

from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops

import pickle

def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def load():
    word_embedding_matrix = __loadStuff("./data/embedding_matrix.p")

    converted_summaries = __loadStuff("./data/converted_summaries.p")
    converted_texts = __loadStuff("./data/converted_texts.p")

    word2ind = __loadStuff("./data/word2ind.p")
    ind2word = __loadStuff("./data/ind2word.p")

    return word_embedding_matrix, converted_summaries, converted_texts, word2ind, ind2word



def text_to_seq(text):
    '''Prepare the text for the model'''

    return [word2ind.get(word, word2ind['<UNK>']) for word in text.split()]





if __name__ == "__main__":

    word_embedding_matrix, converted_summaries, converted_texts, word2ind, ind2word = load()
    
    keep_probability = 0.95
    batch_size = 64
    
    input_sentences = ['the flowers do not get as opened as they look on the picture and the tea does not taste that well very dissapointing']
    '''
    input_sentences=["The coffee tasted great and was at such a good price! I highly recommend this to everyone!", "love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets"]
    '''

    generagte_summary_length =  8

    texts = [text_to_seq(input_sentence) for input_sentence in input_sentences]

    checkpoint = "./sum_model/best_model.ckpt"

    if type(generagte_summary_length) is list:
        if len(input_sentences)!=len(generagte_summary_length):
            raise Exception("[Error] makeSummaries parameter generagte_summary_length must be same length as input_sentences or an integer")
        generagte_summary_length_list = generagte_summary_length
    else:
        generagte_summary_length_list = [generagte_summary_length] * len(texts)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        text_length = loaded_graph.get_tensor_by_name('text_length:0')
        summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        #Multiply by batch_size to match the model's input parameters
        for i, text in enumerate(texts):
            generagte_summary_length = generagte_summary_length_list[i]
            answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                            summary_length: [generagte_summary_length], #summary_length: [np.random.randint(5,8)], 
                                            text_length: [len(text)]*batch_size,
                                            keep_prob: 1.0})[0] 
            # Remove the padding from the summaries
            pad = word2ind["<PAD>"] 
            print('- Review:\n\r {}'.format(input_sentences[i]))
            print('- Summary:\n\r {}\n\r\n\r'.format(" ".join([ind2word[i] for i in answer_logits if i != pad])))
