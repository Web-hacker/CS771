import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC



def label(data):
    
    X = data.iloc[:,:32]
    y0 = data.iloc[:,32]
    y1 = data.iloc[:,33]
    
    return X,y0,y1


def my_fit( X_trn, y0_trn, y1_trn ):

    #X_train,y0_train,y1_train = label(train_data)
    X_train = pd.DataFrame(X_trn)
    y0_train = pd.DataFrame(y0_trn)
    y1_train = pd.DataFrame(y1_trn)
    X_mapped = my_map(X_train)
    
    
    model_0 = LinearSVC(dual='auto')
    model_1 = LinearSVC(dual='auto')
    
    model_0.fit(X_mapped,y0_train)
    model_1.fit(X_mapped,y1_train)
    
    w0 = model_0.coef_
    b0 = model_0.intercept_

    w1 = model_1.coef_
    b1 = model_1.intercept_

	
    return w0, b0, w1, b1


def my_map( X_trn ):

    X = pd.DataFrame(X_trn)
	
    col = list(X.columns)
    col.reverse()
    
    X_reverse = X[col]
    X_reverse = X_reverse.cumprod(axis=1)
    
    X_reverse = pd.concat([X_reverse,X],axis=1)
	
    feat = X_reverse

    return feat
