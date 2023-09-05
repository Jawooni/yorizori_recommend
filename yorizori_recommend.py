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


# In[243]:


def return_recipe_id(recipe_title):
    return md[md['recipe_title']==recipe_title]['recipe_id']

def recommend_recipe(df_svd_preds, user_id, ori_recipe_df, ori_ratings_df, num_recommendations=5):
    
    #현재는 index로 적용이 되어있으므로 user_id - 1을 해야함.
   # user_row_number = user_id - 1 
    target_user = df_user.index(user_id)

    # 최종적으로 만든 pred_df에서 사용자 index에 따라 데이터 정렬 -> 레시피 평점(나중엔 스코어) 높은 순으로 정렬 됌
    sorted_user_predictions = df_svd_preds.iloc[target_user].sort_values(ascending=False)
    
    # 원본 평점 데이터에서 user id에 해당하는 데이터를 뽑아낸다. 
    user_data = ori_ratings_df[ori_ratings_df.user_token_id == user_id]
    #user_data.head(10)
    
    # 위에서 뽑은 user_data와 원본 레시피 데이터를 합친다. 
    user_history = user_data.merge(ori_recipe_df, on = 'recipe_id')
    user_history.sort_values("values",axis=0, ascending=False)
    
    # 원본 레시피들에서 사용자가 본 메뉴를 제외한 데이터를 추출
    recommendations = ori_recipe_df[~ori_recipe_df['recipe_id'].isin(user_history['recipe_id'])]
    
    alpha = pd.DataFrame(sorted_user_predictions).reset_index().rename(columns={'index':'recipe_id',target_user:'Predictions'})
    alpha=alpha.astype({'recipe_id':'int'})
    # 사용자의 평점이 높은 순으로 정렬된 데이터와 합친다. 
    recommendations = recommendations.merge(alpha, on = 'recipe_id',how='left')
                                            
    # 컬럼 이름 바꾸고 정렬해서 return
    recommendations = recommendations.rename(columns = {user_id: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
                      
    
    return user_history, recommendations#alpha


# In[250]:
if __name__ =="__main__":

    recommend_userid = 'abcde123'
    md = pd.read_csv('./yorizori/recipe_yorizori.csv')
    
    df_predict = pd.read_csv('./yorizori_predict_matrix.csv')
    user_info = pd.read_csv('./yorizori_user_values.csv')
    
    f = open('./yorizori_user_index_info.txt','r')
    df_user = f.readlines()
    for i in range(len(df_user)):
        df_user[i]=df_user[i].split("\n")[0]
    

    mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','user_token_id','reference_recipe','level','time'],axis=1)
    



    #target_user = df_user.index(recommend_userid)
    already_rated, predictions = recommend_recipe(df_predict, recommend_userid, mds, user_info, 50) 
    #예측행렬,예측하는유저id, 레시피테이블 데이터프레임, 유저별 로그 데이터프레임, 몇개추천할지
    predictions
    print(predictions['recipe_id'].to_list())

    

# %%
