from keras import models, layers
from corpus_processor import get_processed_corpus, preproc_song
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

def prediction_valid(actual, predicted):
    for i in range(len(actual)):
        if actual[i] != predicted[i]:
            return False
    return True

def model():
    d, l, data, labels = get_processed_corpus()
    border = int(len(data) * 0.5)
    border2 = int(len(data) * 0.7)
    training_set, training_labels = data[:border], labels[:border]
    test_set, test_labels = data[border:border2], labels[border:border2]

    rest_s, rest_l = data[border2:], labels[border2:]
    rest_s_normal, rest_l_normal = d[border2:], labels[border2:]

    model = None
    if LOAD_MODEL_FROM_FILE:
        model = models.load_model('mymodel.h5')
    else:
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(1000,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(training_set, training_labels, epochs=4, batch_size=512)
        model.save('mymodel.h5')



    ###########################
    valid = 0
    invalid = 0

    wr = open("spr.txt", "w+")
    results = model.evaluate(test_set, test_labels)
    #wr.write(str(results))
    for i in range(len(rest_s)):
        wr.write("\n ---- SONG START --- \n")
        wr.write("\n" + rest_s_normal[i])
        prediction = model.predict(np.asarray([rest_s[i]]))
        wr.write("\n" + str(prediction))
        prediction = prediction_modify(prediction[0])
        wr.write("\n===>" + str(prediction))
        wr.write("\n--->" + str(rest_l_normal[i]))



        if(prediction_valid(rest_l[i], prediction)):
            wr.write("\nPREDICTION VALID")
            valid += 1
        else:
            wr.write("\nPREDICTION INVALID")
            invalid += 1
    wr.write("\nVALID: " + str(valid))
    wr.write("\nINVALID: " + str(invalid))

model()