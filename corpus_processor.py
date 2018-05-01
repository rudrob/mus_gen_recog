from os import listdir
from nltk import word_tokenize
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from keras.preprocessing.text import Tokenizer
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from math import log
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from config import *

DIR_METAL = "./corpus/metal/"
DIR_POP = "./corpus/pop/"
DIR_RAP = "./corpus/rap/"
wordnet_lemmatizer = WordNetLemmatizer()
idf = None

def idf_mapping(flat_list):
    mapping = {}
    no_of_songs = len(flat_list)
    for song in flat_list:
        for word in set(word_tokenize(song)):
            if word not in mapping:
                mapping[word] = 1
            else:
                mapping[word] += 1
    for word in mapping:
        mapping[word] = log(no_of_songs/mapping[word])
    return mapping

def get_word_index(songs):
    '''
        word_index is dict of type {'word' : mapping_to_index_according_to_commonality, ... }
    '''
    num_words = INPUT_SHAPE
    del_most_common = DEL_MOST_COMMON
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(songs)
    word_index = {e: i for e, i in tokenizer.word_index.items()
                  if i <= num_words + del_most_common }
    #deleting most common words
    word_index = {k: v for k, v in word_index.items() if v > del_most_common}
    for k in word_index:
        word_index[k] -= del_most_common
    #sequences = tokenizer.texts_to_sequences(songs)
    #one_hot_results = tokenizer.texts_to_matrix(songs, mode='binary')
    return word_index

def text_to_vector(text, word_index):
    '''
        converts text into numpy vector according to word_index
    '''
    vec = np.zeros(len(word_index))
    for word in word_tokenize(text):
        if word in word_index:
            ind = word_index[word] - 1
            vec[ind] += 1
    return vec


def preproc_song(raw_song_text):
    '''
        Preprocesses song from raw textual version to tokenized and lemmatized version
    '''
    raw_song_text = raw_song_text.lower()
    song_text = raw_song_text.strip()
    song_text = ''.join(c for c in song_text if c not in punctuation)
    tokenized = word_tokenize(song_text)
    lemmatized = []
    for token in tokenized:
        lemmatized.append(wordnet_lemmatizer.lemmatize(token))
    song_text = " ".join(lemmatized)
    return song_text

def sample_to_numeric(text, label, word_index):
    vec = text_to_vector(text, word_index)
    label_n = np.asarray([0.0,0.0,0.0])
    if label == "metal":
        label_n = np.asarray([1.0,0.0,0.0])
    elif label == "pop":
        label_n = np.asarray([0.0,1.0,0.0])
    elif label == "rap":
        label_n = np.asarray([0.0,0.0,1.0])
    return vec, label_n

def get_samples(word_index, m_s, p_s, r_s):
    all_len = len(m_s) + len(p_s) + len(r_s)
    flat_list = [item for sublist in [m_s, p_s, r_s] for item in sublist]
    def index_mapping(index):
        assert index >= 0 and index < all_len
        if index < len(m_s):
            return "metal"
        elif index < len(m_s) + len(p_s):
            return "pop"
        else:
            return "rap"

    samples_data = []
    samples_labels = []

    indices = [x for x in range(all_len)]
    random.shuffle(indices)
    for index in indices:
        samples_data.append(flat_list[index])
        samples_labels.append(index_mapping(index))
    numeric_samples = textual_samples_to_numeric_samples(samples_data, samples_labels, word_index)

    return samples_data, samples_labels, numeric_samples[0], numeric_samples[1]

def textual_samples_to_numeric_samples(samples_data, samples_labels, word_index):
    '''
        Takes in textual samples and labels, and word index and
        return those samples in numeric form
    '''
    samples_data_numeric = []
    samples_labels_numeric = []
    for i in range(len(samples_data)):
        numeric = sample_to_numeric(samples_data[i],
                                    samples_labels[i],
                                    word_index)
        samples_data_numeric.append(numeric[0])
        samples_labels_numeric.append(numeric[1])
    return np.asarray(samples_data_numeric), np.asarray(samples_labels_numeric)

def get_processed_corpus():
    metal_song_names = listdir(DIR_METAL)
    pop_song_names = listdir(DIR_POP)
    rap_song_names = listdir(DIR_RAP)

    metal_songs = [preproc_song(open(DIR_METAL + song).read()) for song in metal_song_names]
    pop_songs = [preproc_song(open(DIR_POP + song).read()) for song in pop_song_names]
    rap_songs = [preproc_song(open(DIR_RAP + song).read()) for song in rap_song_names]

    flat_list = [item for sublist in [metal_songs, pop_songs, rap_songs] for item in sublist]
    global idf
    idf = idf_mapping(flat_list)

    words_in_corpus = get_word_index(flat_list)

    return get_samples(words_in_corpus, metal_songs, pop_songs, rap_songs), words_in_corpus




