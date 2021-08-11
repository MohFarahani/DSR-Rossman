import pandas as pd
import numpy as np

def features_by_m(train, store):
    CompMonth_mean = int(store['CompetitionOpenSinceMonth'].mean())
    CompYear_mean = int(store['CompetitionOpenSinceYear'].mean())
    CompDist_mean = int(store['CompetitionDistance'].mean())
    Prom2Week_mean = int(store['Promo2SinceWeek'].mean())
    Prom2Year_mean = int(store['Promo2SinceYear'].mean())
    PromInterval_most = store['PromoInterval'].value_counts().idxmax()
    store.loc[:,'CompetitionOpenSinceMonth'].fillna(value=CompMonth_mean,inplace=True)
    store.loc[:,'CompetitionOpenSinceYear'].fillna(value=CompYear_mean,inplace=True)
    store.loc[:,'CompetitionDistance'].fillna(value=CompYear_mean,inplace=True)
    store.loc[:,'Promo2SinceWeek'].fillna(value=Prom2Week_mean,inplace=True)
    store.loc[:,'Promo2SinceYear'].fillna(value=Prom2Year_mean,inplace=True)
    store.loc[:,'PromoInterval'].fillna(value=PromInterval_most,inplace=True)
    Day_mean = int(train['DayOfWeek'].mean())
    train.loc[:,'DayOfWeek'].fillna(value=Day_mean,inplace=True)
    train.loc[:,'Promo'].fillna(value=0,inplace=True)
    train.loc[:,'SchoolHoliday'].fillna(value=0,inplace=True)
    train.loc[:,'StateHoliday'].fillna(value=0,inplace=True)
    train_full = train.reset_index().merge(store, on='Store', how='left').set_index('Date')
    return train_full

def add_is_promo_month(train_store):
    
    # Indicate whether the month is in promo interval
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    train_store['month_str'] = train_store.index.month.map(month2str)

    def check(row):
        if isinstance(row['PromoInterval'],str) and row['month_str'] in row['PromoInterval']:
            return 1
        else:
            return 0
        
    train_store['IsPromoMonth'] =  train_store.apply(lambda row: check(row),axis=1)
    return train_store

def add_features_by_t(train_store):
    # competition open time (in months)
    train_store['CompetitionOpen'] = 12 * (train_store.index.year - train_store.CompetitionOpenSinceYear) + \
            (train_store.index.month - train_store.CompetitionOpenSinceMonth)

    # Promo open time in months
    train_store['PromoOpen'] = 12 * (train_store.index.year - train_store.Promo2SinceYear) + \
            (train_store.index.weekofyear - train_store.Promo2SinceWeek) / 4.0

    train_store['WeekOfYear'] = train_store.index.weekofyear
    
    train_t = train_store
    train_store['SalePerCustomer'] = train_t['Sales']/train_t['Customers']
    
    train_store = train_store.drop(labels = ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 
                             'Promo2SinceYear', 'Promo2SinceWeek'], axis = 1, errors='ignore')
    
    train_store = add_isPromoMonth(train_store)
    
    return train_store

def add_customer_per_store(train_full):
    
    if('CustomerPerDay' not in train_full.columns):
        cutomer_store = train_full.groupby('Store').agg(cust_st=('Customers','sum'))
        open_store = train_full.groupby('Store').agg(open_st=('Open','count'))
        customer_day_store = cutomer_store["cust_st"]//open_store['open_st']
        temp_df = pd.DataFrame({"CustomerPerDay":customer_day_store})
        train_full = pd.merge(train_full, temp_df, how='inner', on=['Store'])
    
    return train_full

def log_target(train_store):
    """
    Get log of the target as it has a better distribution.
    """
    train_store['Sales'] = train_store['Sales'].map(math.log)
    
    return train_store
