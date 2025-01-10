import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

words_list = list()
bg_to_idx = dict()

def my_fit( words ):

    global words_list
    words_list = sorted(words)
    words_bg = []

    for word in words_list:
        l = [''.join(bg) for bg in zip(word,word[1:])]
        words_bg.append(tuple(sorted(set(l)))[:5])

    unique_words_bg = []

    for bg_tu in words_bg:
        for bg in bg_tu:
            unique_words_bg.append(bg)

    unique_words_bg = sorted(set(unique_words_bg))
    global bg_to_idx
    bg_to_idx = dict((bg , idx) for idx,bg in enumerate(unique_words_bg))
    one_hot_features = np.zeros((len(words_bg),len(unique_words_bg)))

    for idx,bg_tu in enumerate(words_bg):
        for bg in bg_tu:
            idx2 = bg_to_idx[bg]
            one_hot_features[idx,idx2] = 1

    X_train = one_hot_features

    Y_train = np.arange(0,len(words_bg),1)
    
    #model_1 = DecisionTreeClassifier(min_samples_leaf=1)
    #model_1.fit(X_train,Y_train)

    model_2 = RandomForestClassifier()
    model_2.fit(X_train,Y_train)
	
	
    return model_2					



def my_predict( model, bigram_list ):

    guess_list = []
    X_test = np.zeros((1,len(bg_to_idx)))

    for bg in bigram_list:
        idx = bg_to_idx[bg]
        X_test[0,idx] = 1

    pred_list = model.predict(X_test)
    
    for idx in pred_list:
        guess_list.append(words_list[idx])
	
    return guess_list					
