from os import listdir
from nltk import word_tokenize
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from keras.preprocessing.text import Tokenizer
import numpy as np
import random
from nltk.stem import WordNetLemmatizer

DIR_METAL = "./corpus/metal/"
DIR_POP = "./corpus/pop/"
DIR_RAP = "./corpus/rap/"
wordnet_lemmatizer = WordNetLemmatizer()

def get_word_index(songs):
    num_words = 2000
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(songs)
    word_index = {e: i for e, i in tokenizer.word_index.items() if i <= num_words}
    #sequences = tokenizer.texts_to_sequences(songs)
    #one_hot_results = tokenizer.texts_to_matrix(songs, mode='binary')
    return word_index

def text_to_vector(text, word_index):
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
    samples_data_numeric = []
    samples_labels_numeric = []
    indices = [x for x in range(all_len)]
    random.shuffle(indices)
    for index in indices:
        samples_data.append(flat_list[index])
        samples_labels.append(index_mapping(index))
    for i in range(len(samples_data)):
        numeric = sample_to_numeric(samples_data[i],
                                    samples_labels[i],
                                    word_index)
        samples_data_numeric.append(numeric[0])
        samples_labels_numeric.append(numeric[1])
    return samples_data, samples_labels, np.asarray(samples_data_numeric), np.asarray(samples_labels_numeric)


def get_processed_song(song_text):
    song = preproc_song(song_text)

def get_processed_corpus():
    metal_song_names = listdir(DIR_METAL)
    pop_song_names = listdir(DIR_POP)
    rap_song_names = listdir(DIR_RAP)

    metal_songs = [preproc_song(open(DIR_METAL + song).read()) for song in metal_song_names]
    pop_songs = [preproc_song(open(DIR_POP + song).read()) for song in pop_song_names]
    rap_songs = [preproc_song(open(DIR_RAP + song).read()) for song in rap_song_names]

    flat_list = [item for sublist in [metal_songs, pop_songs, rap_songs] for item in sublist]
    '''
    vectorizer = TfidfVectorizer(max_df=0.5)
    X_train_counts = vectorizer.fit_transform(metal_songs)
    id_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    print(id_to_word[100])
    print(X_train_counts[100])
    '''


    words_in_corpus = get_word_index(flat_list)

    return get_samples(words_in_corpus, metal_songs, pop_songs, rap_songs), words_in_corpus

#get_processed_corpus()


