import os
import time
import numpy as np
import re
import html
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import nltk
nltk.download('punkt')


import pickle
def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff
    

def preprocess_sentence(text, keep_most=False):
    """
    Helper function to remove html, unneccessary spaces and punctuation.
    Args:
        text: String.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.

    Returns:
        processed text.
    """
    text = text.lower()
    text = fixup(text)
    text = re.sub(r"<br />", " ", text)
    if keep_most:
        text = re.sub(r"[^a-z0-9%!?.,:()/]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"    ", " ", text)
    text = re.sub(r"   ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = text.strip()
    return text


def fixup(x):
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def preprocess(text, keep_most=False):
    """
    Splits the text into sentences, preprocesses
       and tokenizes each sentence.
    Args:
        text: String. multiple sentences.
        keep_most: Boolean. depending if True or False, we either
                   keep only letters and numbers or also other characters.

    Returns:
        preprocessed and tokenized text.
    """
    tokenized = []
    for sentence in nltk.sent_tokenize(text):
        sentence = preprocess_sentence(sentence, keep_most)
        sentence = nltk.word_tokenize(sentence)
        for token in sentence:
            tokenized.append(token)
    return tokenized


def preprocess_texts_and_summaries(texts, summaries, keep_most=False):
    """iterates given list of texts and given list of summaries and tokenizes every
       text using the tokenize_review() function.
       apart from that we count up all the words in the texts and summaries.
       returns: - processed texts
                - processed summaries
                - array containing all the unique words together with their counts
                  sorted by counts.
    """

    start_time = time.time()
    processed_texts = []
    processed_summaries = []
    words = []

    for text in texts:
        text = preprocess(text, keep_most)
        for word in text:
            words.append(word)
        processed_texts.append(text)
    for summary in summaries:
        summary = preprocess(summary, keep_most)
        for word in summary:
            words.append(word)
        processed_summaries.append(summary)

    words_counted = Counter(words).most_common()
    print('Processing Time: ', time.time() - start_time)

    return processed_texts, processed_summaries, words_counted


def create_word_indx_dicts(words_counted, specials=None, min_occurences=0):
    """ creates lookup dicts from word to index and back.
        returns the lookup dicts and an array of words that were not used,
        due to rare occurence.
    """
    ignore_words = []
    word2ind = {}
    ind2word = {}
    i = 0

    if specials is not None:
        for sp in specials:
            word2ind[sp] = i
            ind2word[i] = sp
            i += 1

    for (word, count) in words_counted:
        if count >= min_occurences:
            word2ind[word] = i
            ind2word[i] = word
            i += 1
        else:
            ignore_words.append(word)

    return word2ind, ind2word, ignore_words


def convert_sentence(sent, word2ind):
    """ converts the given sent to int values corresponding to the given word2ind"""
    indx = []
    unknown_words = []

    for word in sent:
        if word in word2ind.keys():
            indx.append(int(word2ind[word]))
        else:
            indx.append(int(word2ind['<UNK>']))
            unknown_words.append(word)

    return indx, unknown_words


def convert_text_to_indx(input, word2ind, eos=False, sos=False):
    """ converts the given all texts input to int values corresponding to the given word2ind"""
    converted_input = []
    all_unknown_words = set()

    for inp in input:
        converted_inp, unknown_words = convert_sentence(inp, word2ind)
        if sos:
            converted_inp.insert(0, word2ind['<SOS>'])
        if eos:
            converted_inp.append(word2ind['<EOS>'])
        converted_input.append(converted_inp)
        all_unknown_words.update(unknown_words)

    return converted_input, all_unknown_words


def convert_indx_to_text(indx, ind2word, preprocess=False):
    """ convert the given indexes back to text """
    words = [ind2word[word] for word in indx]
    return words


def load_pretrained_embeddings(path):
    """loads pretrained embeddings. stores each embedding in a dictionary with its corresponding word"""
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding_vector = np.array(values[1:], dtype='float32')
            embeddings[word] = embedding_vector
    
    return embeddings


def create_and_save_embedding_matrix(word2ind, pretrained_embeddings_path, save_path, embedding_dim=300):
    """creates embedding matrix for each word in word2ind. if that words is in
       pretrained_embeddings, that vector is used. otherwise initialized randomly.
    """
    pretrained_embeddings = load_pretrained_embeddings(pretrained_embeddings_path)
    embedding_matrix = np.zeros((len(word2ind), embedding_dim), dtype=np.float32)
    for word, i in word2ind.items():
        if word in pretrained_embeddings.keys(): 
            embedding_matrix[i] = pretrained_embeddings[word]
        else:
            embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embedding_matrix[i] = embedding
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, embedding_matrix)
    
    return np.array(embedding_matrix)




def pad_sentence_batch(sentence_batch, word2ind):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    
    return [sentence + [word2ind['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]



def get_batches(word2ind, summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch, word2ind))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch, word2ind))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths



def reset_graph(seed=97):
    """helper function to reset the default graph. this often
       comes handy when using jupyter noteboooks.
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)




if __name__ == "__main__":
    file_path = './Reviews.csv'
    data = pd.read_csv(file_path)

    data.dropna(subset=['Summary'],inplace = True)
    data = data[['Summary', 'Text']]


    raw_texts = []
    raw_summaries = []

    for text, summary in zip(data.Text, data.Summary):
        if 20 < len(text) < 300:
            raw_texts.append(text)
            raw_summaries.append(summary)

    # the function gives us the option to keep_most of the characters inisde the texts and summaries, meaning
    # punctuation, question marks, slashes...
    # or we can set it to False, meaning we only want to keep letters and numbers like here.
    processed_texts, processed_summaries, words_counted = preprocess_texts_and_summaries(
                raw_texts,
                raw_summaries,
                keep_most=False)

    specials = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    word2ind, ind2word,  ignore_words = create_word_indx_dicts(words_counted, specials=specials)

    print(len(word2ind), len(ind2word), len(ignore_words))


    # the embeddings from tf_hub. 
    # embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
    embed = hub.Module("https://tfhub.dev/google/Wiki-words-250/1")
    emb = embed([key for key in word2ind.keys()])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embedding_matrix = sess.run(emb)

    print(embedding_matrix.shape)
    np.save('./tf_hub_embedding.npy', embedding_matrix)

    # converts words in texts and summaries to indices
    # it looks like we have to set eos here to False
    converted_texts, unknown_words_in_texts = convert_text_to_indx(processed_texts, word2ind, eos=True, sos=False)

    converted_summaries, unknown_words_in_summaries = convert_text_to_indx(processed_summaries, word2ind, eos=False, sos=False)



    __pickleStuff("./data/embedding_matrix.p", embedding_matrix)
    __pickleStuff("./data/converted_summaries.p", converted_summaries)
    __pickleStuff("./data/converted_texts.p", converted_texts)
    __pickleStuff("./data/word2ind.p",word2ind)
    __pickleStuff("./data/ind2word.p",ind2word)