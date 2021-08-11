import pandas as pd
import numpy as np
import sys
import pickle
import xgboost as xgb
import category_encoders as ce
import math
import RossmanFeatures


PATH_STORE = 'modified_store.csv'
PATH_MODEL = 'xgb.pickle.dat'

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

def evaluate(path_X_csv,path_y_csv):
    train = pd.read_csv(path_X_csv)
    y_real = pd.read_csv(path_y_csv)

    if 'Open' in train.columns:
        mask_open = train['Open']==1
        train = train[mask_open]
        y_real = y_real[mask_open]

    y_predict = predict(train)

    RMSPE = metric(y_predict, y_real)
    print("RMSPE: ",RMSPE)
    return RMSPE

def clean(df):
    if 'StateHoliday' in df.columns:
        df.loc[df['StateHoliday']==0,'StateHoliday']='0'
        df.loc[df['StateHoliday']=='0','StateHoliday']='0'

def fillna(df):
    def fillna_mean(df,columns):
        for col in columns:
            if col in df.columns:
                mean_value = int(df[col].mean())
                df.loc[:,col].fillna(value=mean_value,inplace=True)
        return df

    def fillna_most(df,columns):
        for col in columns:
            if col in df.columns:
                most_value = df[col].value_counts().idxmax()
                df.loc[:,col].fillna(value=most_value,inplace=True)
        return df
    columns_mean = ['DayOfWeek','Customers']
    df = fillna_mean(df,columns_mean)

    columns_most = ['Promo','SchoolHoliday','StateHoliday']
    df = fillna_most(df,columns_most)


def target_encoding(X, y):
    
    ce_te = ce.TargetEncoder(cols=['Store', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval','month_str'])
    ce_te.fit(X,y)
    X = ce_te.transform(X)
    
    return X

def new_features(df):
    df = RossmanFeatures.features_by_m(df)
    df = RossmanFeatures.features_by_t(df)
    df = RossmanFeatures.add_is_promo_month(df)
    df = RossmanFeatures.add_features_by_t(df)
    df = RossmanFeatures.add_customer_per_store(df)
    df = RossmanFeatures.log_target(df)

    

def drop_columns(df,columns):
    for col in columns:
        if col in df.columns:
            df.drop(columns=[col],inplace=True)
    return df

def predict(train):
    store = pd.read_csv(PATH_STORE)
    train = clean(train)
    train = fillna(train)
    train_full = pd.merge(train,store, on='Store', how='inner').set_index='Date'
    train_full = encoding(train_full)
    train_full = new_features(train_full)
    drop_columns_list = ['Store','Customer','Date','Open']
    train_full = train_full.drop(labels=drop_columns_list, axis=1, errors='ignore')
    train_full.dropna(inplace=True)
    xgb = pickle.load(open(PATH_MODEL, "rb"))
    return xgb.predict(train)

if __name__ == "__main__":
    path_X_csv,path_y_csv = sys.argv[1] , sys.argv[2]
    RMSPE = evaluate(path_X_csv,path_y_csv)