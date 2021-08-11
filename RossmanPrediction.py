import pandas as pd
import numpy as np
import sys

PATH_STORE = 'modified_store.csv'
PATH_MODEL = 'xgb.pickle'

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def evaluate(path_X_csv,path_y_csv):
    train = pd.read_csv(path_X_csv)
    mask_open = train['Open']==1
    train = train[mask_open]
    y_real = pd.read_csv(path_y_csv)
    y_real = y_real[mask_open]
    y_predict = predict(train)
    RMSPE = metric(y_predict, y_real)
    print("RMSPE: ",RMSPE)
    return RMSPE

def clean(df):
    pass

def fillna(df):
    pass

def encoding(df):
    pass

def new_features(df):
    pass

def predict(train):
    store = pd.read_csv(PATH_STORE)
    train = clean(train)
    train = fillna(train)
    train_full = pd.merge(train,store, on='Store', how='inner')
    train_full = encoding(train_full)
    train_full = new_features(train_full)
    train_full.dropna(inplace=True)
    xgb = loadXGB
    return xgb.predict(train)

if __name__ == "__main__":
    path_X_csv,path_y_csv = sys.argv[1] , sys.argv[2]
    RMSPE = evaluate(path_X_csv,path_y_csv)