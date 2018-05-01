from keras import models, layers
from corpus_processor import get_processed_corpus, preproc_song, sample_to_numeric
import numpy as np
from config import *


def prediction_modify(pred):
    pred_mod = np.zeros(len(pred))
    indmax = 0
    for i in range(len(pred)):
        if pred[i] > pred[indmax]:
            indmax = i
    pred_mod[indmax] = 1.0
    return pred_mod


def numeric_prediction_to_textual(pred):
    if pred[0] == 1:
        return "metal"
    elif pred[1] == 1:
        return "pop"
    elif pred[2] == 1:
        return "rap"


def prediction_valid(actual, predicted):
    for i in range(len(actual)):
        if actual[i] != predicted[i]:
            return False
    return True


def get_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(INPUT_SHAPE,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    data_bundle, word_index = get_processed_corpus()

    ''' print words sorted by commonality
    words = [0 for x in range(2001)]
    for z in word_index:
        words[word_index[z]] = z
    print(words)
    '''

    d, l, data, labels = data_bundle[0], data_bundle[1], data_bundle[2], data_bundle[3]
    border = int(len(data) * 0.65)
    border2 = int(len(data) * 0.8)
    training_set, training_labels = data[:border], labels[:border]
    test_set, test_labels = data[border:border2], labels[border:border2]

    rest_s, rest_l = data[border2:border2 + 200], labels[border2:border2 + 200]
    rest_s_normal, rest_l_normal = d[border2:border2 + 200], labels[border2:border2 + 200]
    print("Training set len: ", len(training_set))
    print("Testing set len: ", len(test_set))
    print("Rest len: ", len(rest_s))
    model = None
    if LOAD_MODEL_FROM_FILE:
        model = models.load_model('mymodel.h5')
    else:
        model = get_model()
        model.fit(training_set, training_labels, epochs=EPOCHS_NO, batch_size=BATCH_SIZE)
        model.save('mymodel.h5')


    if not TEST_SONG_FROM_FILE:
        valid = 0
        invalid = 0

        wr = open("other/spr.txt", "w+")
        results = model.evaluate(test_set, test_labels)
        # wr.write(str(results))
        for i in range(len(rest_s)):
            wr.write("\n ---- SONG START --- \n")
            wr.write("\n" + rest_s_normal[i][:10])
            prediction = model.predict(np.asarray([rest_s[i]]))
            wr.write("\n" + str(prediction))
            prediction = prediction_modify(prediction[0])
            wr.write("\n===>" + str(prediction))
            wr.write("\n--->" + str(rest_l_normal[i]))

            if (prediction_valid(rest_l[i], prediction)):
                wr.write("\nPREDICTION VALID")
                valid += 1
            else:
                wr.write("\nPREDICTION INVALID")
                invalid += 1
        wr.write("\nVALID: " + str(valid))
        wr.write("\nINVALID: " + str(invalid))
        wr.write("\nVALID PERCENT: " + str(100 * (valid / (valid + invalid))))
    else:
        test_song_text = open(TEST_SONG_FROM_FILE_NAME).read()
        test_song_text = preproc_song(test_song_text)
        numeric = sample_to_numeric(test_song_text, "pop", word_index)
        prediction = model.predict(np.asarray([numeric[0]]))
        prediction = prediction_modify(prediction[0])
        print("Predicted: ", numeric_prediction_to_textual(prediction))


main()
