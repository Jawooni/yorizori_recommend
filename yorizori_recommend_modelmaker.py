#!/usr/bin/env python
# coding: utf-8

# In[223]:


import pandas as pd #csv파일 읽어오는 모듈
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse.linalg import svds
from ast import literal_eval

from surprise import Reader, Dataset, SVD, accuracy


import warnings; warnings.simplefilter('ignore')



# In[310]:


def return_recipe_id(recipe_title):
    return md[md['recipe_title']==recipe_title]['recipe_id']

def renewal_view_log_info(user_view_info):
    tmp=user_view_info.groupby('user_token_id')['view_recipe_id'].value_counts().to_frame()

    tmp2=tmp.swaplevel('user_token_id','view_recipe_id')
    tmp2.rename(columns={'view_recipe_id':'count'},inplace=True)

    user_view_log=tmp2.reset_index()
    user_view_log.rename(columns={'view_recipe_id':'recipe_id'},inplace=True)

    return user_view_log

def renewal_user_rate_info(user_rate_info):
    a=user_rate_info.groupby(['user_token_id','recipe_id'])['scope'].mean().to_frame()
    b=a.swaplevel('user_token_id','recipe_id')
#b
#tmp=user_view_log2.groupby('user_token_id')['view_recipe_id'].value_counts().to_frame()
    user_rate_log=b.reset_index()
    return user_rate_log


def make_pivot_table(user_view_info,user_rate_info,index,column,value):

    user_rate=renewal_user_rate_info(user_rate_info)
    user_view=renewal_view_log_info(user_view_info)
    thisplus=pd.merge(user_rate,user_view,how='right').fillna(0)
    thisplus['scope'] = (thisplus['scope'] - thisplus['scope'].min())/(thisplus['scope'].max()-thisplus['scope'].min())
    thisplus['count'] = (thisplus['count'] - thisplus['count'].min())/(thisplus['count'].max()-thisplus['count'].min())
    thisplus['values'] = (thisplus['scope']+thisplus['count']).fillna(0)
    
    df_user_recipe_ratings = thisplus.pivot( #사용자 평점 정보를 피벗테이블 형식으로 바꾸기
    index=index,
    columns=column,
    values=value).fillna(0)
    
    nb=df_user_recipe_ratings.index.to_list()

    return df_user_recipe_ratings , thisplus,nb


def filter_df(df_user_recipe_ratings):
    mt=df_user_recipe_ratings.values

    user_rating_mean = np.mean(mt, axis=1)

    matrix_user_mean = mt-user_rating_mean.reshape(-1,1)

    U,sigma, Vt = svds(matrix_user_mean, k=6)

    sigma = np.diag(sigma)

    svd_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt)+user_rating_mean.reshape(-1,1)
    df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_recipe_ratings.columns)

    return df_svd_preds
# In[250]:
if __name__ =="__main__":

    recommend_userid = 'abcde123'
    md = pd.read_csv('./data/recipe_yorizori.csv')
    user_rate_info2 = pd.read_csv('./data/test2.csv') #해당 유저의 별점과 로그, 최근 조회레시피 검색기록 불러오기
    user_search_ingredient = pd.read_csv('./data/search_ingredient_log.csv')
    user_search_recipe =  pd.read_csv('./data/search_recipe_log.csv')
    user_view_log =  pd.read_csv('./data/test1.csv')

    mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','level','time'],axis=1)
    user_rate_info2=user_rate_info2.drop('Unnamed: 3',axis=1)
    user_view_log2=user_view_log.drop(['view_log_id','created_time','updated_time'],axis=1)

    df_user,thisplus,nb=make_pivot_table(user_view_log2,user_rate_info2,'user_token_id','recipe_id','values')

    df_svd_pred=filter_df(df_user)

    df_svd_pred.to_csv('./yorizori_predict_matrix.csv', index=False)
    thisplus.to_csv('./yorizori_user_values.csv',index=False)
    with open('yorizori_user_index_info.txt','w',encoding='UTF-8') as f:
        for name in nb:
            f.write(name+'\n')

    
