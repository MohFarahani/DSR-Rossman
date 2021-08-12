# Packages
import pandas as pd
import numpy as np
import sys
import pickle
import math
import xgboost as xgb
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split


# PATH to Input data
PATH_TRAIN = "data/train.csv"
PATH_STORE = "data/store.csv"
PATH_STORE_MODIFIED = "data/store_modified.csv"
MODEL_NAME = "XGBoost.txt"
LABEL_ENCODE="LabelEncode"
TARGET_ENCODE="TargetEncode"

# Define RMSPE for evaluation
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

class Rossman():
    
    def __init__(self,train_path=None,store_path=None,store_modified_path=None):
        
        self.train_path = train_path
        self.store_path = store_path
        self.store_modified_path = store_modified_path
        self.train = None
        self.store = None
        self.model = None
    
    
    def read_data(self,path_train,path_store):
        
        self.train = pd.read_csv(path_train, parse_dates=True, low_memory = False, index_col=None)
        self.train['Date'] = pd.to_datetime(self.train["Date"])
        self.store = pd.read_csv(path_store, low_memory=False)
        return self.train,self.store
                                 
    def clean(self,df):
        df=df[df['Sales']>0]
        if 'StateHoliday' in df.columns:
            df.loc[df['StateHoliday']==0,'StateHoliday']='0'
            df.loc[df['StateHoliday']=='0','StateHoliday']='0'
        return df
    
                                 
    def fillna(self,df,columns_mean,columns_most):
                                 
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
        
        df = fillna_mean(df,columns_mean)
        df = fillna_most(df,columns_most)

        return df 
    
                                 
    def fillna_train(self,df_train):
                                 
        columns_mean = ['DayOfWeek','Customers']
        columns_most = ['Promo','SchoolHoliday','StateHoliday']
        df_train = self.fillna(df_train,columns_mean,columns_most)
        return df_train
    
                                 
    def fillna_store(self,df_store):
        columns_mean = ['CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear',
                'CompetitionDistance',
                'Promo2SinceWeek',
                'Promo2SinceYear'
                 ] 
        columns_most = ['PromoInterval']
        df_store = self.fillna(df_store,columns_mean,columns_most)
        return df_store

                                 
    def merge_train_store(self,train,store):
                                 
        train_full = train.merge(store, on='Store', how='inner')
        return train_full
    
                                 
    def add_features(self,train_full,store,UPDATE=False):
        
        if UPDATE== False:
            if('CustomerPerDay' not in train_full.columns):
                cutomer_store = train_full.groupby('Store').agg(cust_st=('Customers','sum'))
                open_store = train_full.groupby('Store').agg(open_st=('Open','count'))
                customer_day_store = cutomer_store["cust_st"]//open_store['open_st']
                temp_df = pd.DataFrame({"CustomerPerDay":customer_day_store})
                train_full = pd.merge(train_full, temp_df, how='inner', on=['Store'])
                store = pd.merge(store, temp_df, how='inner', on=['Store'])
        
        train_full['CompetitionOpen'] = 12 * (train_full.loc[:,'Date'].dt.year -
                                              train_full.CompetitionOpenSinceYear)+\
                                        (train_full.loc[:,'Date'].dt.month -
                                         train_full.CompetitionOpenSinceMonth)

        # Promo open time in months
        train_full['PromoOpen'] = 12 * (train_full.loc[:,'Date'].dt.year - train_full.Promo2SinceYear)+\
                              (train_full.loc[:,'Date'].dt.weekofyear - train_full.Promo2SinceWeek)/ 4.0

        train_full['WeekOfYear'] = train_full.loc[:,'Date'].dt.weekofyear

        
        return train_full,store
    

    def encode_choice(self):
                                 
        encode_dict = {}
        encode_dict['OneHot'] = ['StoreType','Assortment','PromoInterval']
        encode_dict['Label'] = ['StateHoliday']
        encode_dict['Freq'] = [] 
        encode_dict['Target'] = ['Store'] 
        return encode_dict


    def encoding(self,train_full,TRAIN=True):
                                 
        encode_dict = self.encode_choice()
        for key,value in encode_dict.items():
            
            if key=='OneHot':
                for col in value:
                    if col in train_full.columns:
                        train_full = pd.get_dummies(train_full, columns = [col])
                        
            if key=='Label':
                if TRAIN==True:
                    le= LabelEncoder()
                    for col in value:
                        if col in train_full.columns: 
                           train_full[col] = le.fit_transform(train_full[col])
                    LABEL_FILE = open(LABEL_ENCODE,"wb")
                    pickle.dump(le,LABEL_FILE)
                    LABEL_FILE.close()
                else:
                    LABEL_FILE = open(LABEL_ENCODE,"rb")
                    le = pickle.load(LABEL_FILE)
                    LABEL_FILE.close()
                    for col in value:
                        if col in train_full.columns: 
                           train_full[col] = le.fit(train_full[col])
                    
                    
            elif key=='Freq':
                for col in value:
                    if col in train_full.columns:
                        freq = train_full.groupby(col).size()/len(train_full)
                        train_full.loc[:,col+'_freq'] = train_full.loc[:,col].map(freq)
            if key=='Target': 
                if TRAIN==True:
                    te= TargetEncoder(cols=value)
                    te.fit_transform(train_full,train_full['Sales'])
                    TARGET_FILE = open(TARGET_ENCODE,"wb")
                    pickle.dump(le,TARGET_FILE)
                    TARGET_FILE.close()
                else:
                    TARGET_FILE = open(TARGET_ENCODE,"rb")
                    te = pickle.load(TARGET_FILE)
                    TARGET_FILE.close()
                    te.fit(train_full)
                
        return train_full
    
                                 
    def drop_columns(self,train_full):
                                 
        columns = ['Store','Customer','Date','Open',
                   'CompetitionOpenSinceMonth',
                   'CompetitionOpenSinceYear', 
                   'Promo2SinceYear', 'Promo2SinceWeek']
        train_full.drop(columns = columns, inplace=True, errors='ignore')
        return train_full
                                 
    def X_y(self,train_full): 
                                 
        X = train_full.drop(columns=['Sales'])
        y = train_full['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        return X_train, X_test, y_train, y_test
     
                                 
    def model_xgb(self, X_train, X_test, y_train, y_test):
        def log_target(y_train, y_test):
            """
            Get log of the target as it has a better distribution.
            """
            y_train_log = np.log2(y_train)
            y_test_log =  np.log2(y_test)

            return y_train, y_test
        
        def rmspe(y, yhat):
            return np.sqrt(np.mean((yhat/y-1) ** 2))

        def rmspe_xg(yhat, y):
            y = np.expm1(y.get_label())
            yhat = np.expm1(yhat)
            return "rmspe", rmspe(y,yhat)       
        
        y_train, y_test = log_target(y_train, y_test)
        params = {"objective": "reg:linear", # for linear regression
          "booster" : "gbtree",   # use tree based models 
          "eta": 0.03,   # learning rate
          "max_depth": 10,    # maximum depth of a tree
          "subsample": 0.9,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
          "silent": 1,   # silent mode
          "seed": 10   # Random number seed
          }
        num_boost_round = 1500

        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_test, y_test)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(params, dtrain, num_boost_round, evals=watchlist, 
                          early_stopping_rounds= 100, feval=rmspe_xg, verbose_eval=True)
        return model
        
    def xgb_simple(self, X_train, X_test, y_train, y_test):       
        model = xgb.XGBRegressor(max_depth=10,n_estimators=200)
        # fit model
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        RMSPE = metric(y_predict, y_test.to_numpy())
        print("RMSPE: ",RMSPE)
        return model,RMSPE
    
    def training(self,train_path,store_path,path_store_modified,model_name):
        train,store = self.read_data(train_path,store_path)
        train = self.clean(train)
        train = self.fillna_train(train)
        store = self.fillna_store(store)
        train_full = self.merge_train_store(train,store)
        train_full,store = self.add_features(train_full,store)
        train_full = self.encoding(train_full)
        train_full = self.drop_columns(train_full)
        X_train, X_test, y_train, y_test = self.X_y(train_full)
        #self.model = self.model_xgb(X_train, X_test, y_train, y_test)
        self.model = self.xgb_simple(X_train, X_test, y_train, y_test)
        store.to_csv(path_store_modified)
        self.store_modified_path = path_store_modified
        self.model.save_model(model_name)
        
        
    def testing(self,path_test,path_store_modified,MODEL_NAME):
        test,store = self.read_data(path_test,path_store_modified)
        test = self.clean(test)
        test = self.fillna_train(test)
        test_full = self.merge_train_store(test,store)
        test_full,_ = self.add_features(test_full,store,UPDATE=True)
        test_full = self.encoding(test_full,TRAIN=False)
        test_full = self.drop_columns(test_full)
        test_full.dropna()
        X = test_full.drop(columns=['Sales'])
        y = test_full['Sales']
        self.model = xgb.XGBRegressor()
        self.model.load_model(MODEL_NAME)
        y_predict = self.model.predict(X)
        RMSPE = metric(y_predict, y.to_numpy())
        return RMSPE
      

if __name__ == "__main__":
    rossman = Rossman()
    RMSPE = rossman.training(PATH_TRAIN,PATH_STORE,PATH_STORE_MODIFIED,MODEL_NAME)   
    print("RMSPE: ",RMSPE)          