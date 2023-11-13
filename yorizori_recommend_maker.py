

# In[243]:


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

#from surprise import Reader, Dataset, SVD, accuracy


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

def merge_category_view_log(user_view_info,recipe_category):
    user_view=renewal_view_log_info(user_view_info)
    user_view_info.columns=['user_token_id','recipe_id']
    
    view_category = pd.merge(user_view_info,recipe_category)
    
    
    tmp=view_category.groupby('user_token_id')['recipe_id'].value_counts().to_frame()
    tmps = view_category.groupby('user_token_id')['category'].value_counts().to_frame()

    tmp2=tmp.swaplevel('user_token_id','recipe_id')
    tmp2.rename(columns={'recipe_id':'recipe_view_count'},inplace=True)

    tmps2=tmps.swaplevel('user_token_id','category')
    tmps2.rename(columns={'category':'category_view_count'},inplace=True)

    user_recipe_view_log=tmp2.reset_index()
    user_category_view_log=tmps2.reset_index()
    
    tts = user_category_view_log
    tts['category_value']=(tts['category_view_count'] - tts['category_view_count'].min())/(tts['category_view_count'].max()-tts['category_view_count'].min())
    
    thisplus2=pd.merge(tts,user_recipe_view_log,how='right').fillna(0)
    
    return thisplus2
def renewal_user_rate_info(user_rate_info_):
    a=user_rate_info_.groupby(['user_token_id','recipe_id'])['star_count'].mean().to_frame()
    b=a.swaplevel('user_token_id','recipe_id')
#b
#tmp=user_view_log2.groupby('user_token_id')['view_recipe_id'].value_counts().to_frame()
    user_rate_log=b.reset_index()
    return user_rate_log


def make_pivot_table(user_info,user_view_info,user_rate_info,recipe_category,index,column,value,age,gender):
    user_infos = user_info[user_info['gender'].isin([gender])&user_info['age'].isin([age])]
    
    user_rate=renewal_user_rate_info(user_rate_info)
    user_view=merge_category_view_log(user_view_info,recipe_category)
    user_view=user_view.drop(['category'],axis=1)
    user_view['recipe_view_count'] = (user_view['recipe_view_count'] - user_view['recipe_view_count'].min())/(user_view['recipe_view_count'].max()-user_view['recipe_view_count'].min())
    
    userview = user_view.drop(['category_view_count','category_value'],axis=1)
    userview = userview.drop_duplicates(subset=['recipe_id','user_token_id'])
    
    a=user_view.groupby(['user_token_id','recipe_id'])['category_value'].mean().to_frame()
    b=a.swaplevel('user_token_id','recipe_id')
    b=b.reset_index()
    
    userview_value_tmp = pd.merge(b,userview,how='left')
    
    userview_value = pd.merge(user_infos,userview_value_tmp,how='left').fillna(0,downcast='infer')
    thisplus=pd.merge(user_rate,userview_value,how='right').fillna(0)
    thisplus['star_count'] = (thisplus['star_count'])/(thisplus['star_count'].max())
    #starcount 적용할 수는 있는데 지금 user엔 있는데 viewlog엔 없는사람들이 많아서 잠깐 0으로
    thisplus['values'] = 100*((thisplus['star_count']*3)+(thisplus['recipe_view_count']*5)+(thisplus['category_value']*30)).fillna(0.0000)
    
    thisplus.drop(['category_value'],axis=1)
    #c=pd.merge(thisplus,b,how='left',on=['recipe_id','user_token_id']).fillna(0)
    #c['values'] = ((c['star_count']*3)+(c['recipe_view_count']*5)+(c['category_value_y']*10)).fillna(0.0000)
    df_user_recipe_ratings = thisplus.pivot( #사용자 평점 정보를 피벗테이블 형식으로 바꾸기
    index=index,
    columns=column,
    values=value).fillna(0)
    
    nb=df_user_recipe_ratings.index.to_list()

    return df_user_recipe_ratings ,thisplus,nb,userview_value
def make_pivot_table_all(user_info,user_view_info,user_rate_info,recipe_category,index,column,value):
    user_rate=renewal_user_rate_info(user_rate_info)
    user_view=merge_category_view_log(user_view_info,recipe_category)
    user_view=user_view.drop(['category'],axis=1)
    user_view['recipe_view_count'] = (user_view['recipe_view_count'] - user_view['recipe_view_count'].min())/(user_view['recipe_view_count'].max()-user_view['recipe_view_count'].min())
    
    userview = user_view.drop(['category_view_count','category_value'],axis=1)
    userview=userview.drop_duplicates(subset=['recipe_id','user_token_id'])
    
    a=user_view.groupby(['user_token_id','recipe_id'])['category_value'].mean().to_frame()
    b=a.swaplevel('user_token_id','recipe_id')
    b=b.reset_index()
    
    userview_value_tmp = pd.merge(b,userview,how='left')
    
    userview_value = pd.merge(user_info,userview_value_tmp,how='left').fillna(0,downcast='infer')
    thisplus=pd.merge(user_rate,userview_value,how='right').fillna(0)
    thisplus['star_count'] = (thisplus['star_count'])/(thisplus['star_count'].max())
    #starcount 적용할 수는 있는데 지금 user엔 있는데 viewlog엔 없는사람들이 많아서 잠깐 0으로
    thisplus['values'] = ((thisplus['star_count']*3)+(thisplus['recipe_view_count']*5)+(thisplus['category_value']*10)).fillna(0.0000)
    
    thisplus.drop(['category_value'],axis=1)
    #c=pd.merge(thisplus,b,how='left',on=['recipe_id','user_token_id']).fillna(0)
    #c['values'] = ((c['star_count']*3)+(c['recipe_view_count']*5)+(c['category_value_y']*10)).fillna(0.0000)
    df_user_recipe_ratings = thisplus.pivot( #사용자 평점 정보를 피벗테이블 형식으로 바꾸기
    index=index,
    columns=column,
    values=value).fillna(0)
    
    nb=df_user_recipe_ratings.index.to_list()

    return df_user_recipe_ratings ,thisplus,nb
#df_user_recipe_ratings , 
def filter_df(df_user_recipe_ratings):
    mt=df_user_recipe_ratings.values

    user_rating_mean = np.mean(mt, axis=1)

    matrix_user_mean = mt-user_rating_mean.reshape(-1,1)
    
    num = matrix_user_mean.shape[0]-1
    
    U,sigma, Vt = svds(matrix_user_mean, k=num)

    sigma = np.diag(sigma)

    svd_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt)+user_rating_mean.reshape(-1,1)
    df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_recipe_ratings.columns)

    return df_svd_preds


def recommend_maker():
    md = pd.read_csv('./yorizori/recipe_yorizori.csv')
    user_rate_data = pd.read_csv('./yorizori/user_comment.csv').drop(['comment_id','created_time','updated_time','comment'],axis=1) #해당 유저의 별점과 로그, 최근 조회레시피 검색기록 불러오기
    user_view_data =  pd.read_csv('./yorizori/user_view_recipe_log.csv').drop(['view_log_id','created_time'],axis=1)
    recipe_category_data = pd.read_csv('./yorizori/recipe_category.csv').drop(['category_id'],axis=1)
    user_info_data = pd.read_csv('./yorizori/yorizori_user.csv').drop(['created_time','updated_time','image_address','nickname','oauth_division'],axis=1)
    mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','level','time'],axis=1)

    user_rate_data.columns =['star_count','user_token_id','recipe_id']
    
    
    df_user,thisplus,nb = make_pivot_table_all(user_info_data,user_view_data,user_rate_data,recipe_category_data,'user_token_id','recipe_id','values')
    df_svd_pred_all=filter_df(df_user)
    filepath = './yorizori_predict/yorizori_predict_matrix.csv'
    df_svd_pred_all.to_csv(filepath, index=False)
    thisplus.to_csv('./yorizori_predict/yorizori_user_values.csv',index=False)
    with open('./yorizori_predict/yorizori_user_index_info.txt','w',encoding='UTF-8') as f:
        for name in nb:
            f.write(name+'\n')
    


    
    man_age_pool = user_info_data[user_info_data['gender']=="남자"]['age'].unique()
    woman_age_pool = user_info_data[user_info_data['gender']=="여자"]['age'].unique()
    
    
    for age in man_age_pool:
        md = pd.read_csv('./yorizori/recipe_yorizori.csv')
        user_rate_data = pd.read_csv('./yorizori/user_comment.csv').drop(['comment_id','created_time','updated_time','comment'],axis=1) #해당 유저의 별점과 로그, 최근 조회레시피 검색기록 불러오기
        user_view_data =  pd.read_csv('./yorizori/user_view_recipe_log.csv').drop(['view_log_id','created_time'],axis=1)
        recipe_category_data = pd.read_csv('./yorizori/recipe_category.csv').drop(['category_id'],axis=1)
        user_info_data = pd.read_csv('./yorizori/yorizori_user.csv').drop(['created_time','updated_time','image_address','nickname','oauth_division'],axis=1)
        mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','level','time'],axis=1)

        user_rate_data.columns =['star_count','user_token_id','recipe_id']
        print(age,"남자")
            
        
        df_user2,thisplus2,nb2,userv2=make_pivot_table(user_info_data,user_view_data,user_rate_data,recipe_category_data,'user_token_id','recipe_id','values',age,"남자")
    
    #df_user_recipe_ratings , thisplus,nb
        if(df_user2.shape[0]<3):
            df_svd_pred = df_svd_pred_all
        else :
            df_svd_pred=filter_df(df_user2)
            filepath = './yorizori_predict/yorizori_predict_matrix'+age+'_남자.csv'
            df_svd_pred.to_csv(filepath, index=False)
        thisplus2.to_csv('./yorizori_predict/yorizori_user_values'+age+'_남자.csv',index=False)
        with open('./yorizori_predict/yorizori_user_index_info'+age+'_남자.txt','w',encoding='UTF-8') as f:
            for name in nb2:
                f.write(name+'\n')
                    
    for age1 in woman_age_pool:
        md = pd.read_csv('./yorizori/recipe_yorizori.csv')
        user_rate_data = pd.read_csv('./yorizori/user_comment.csv').drop(['comment_id','created_time','updated_time','comment'],axis=1) #해당 유저의 별점과 로그, 최근 조회레시피 검색기록 불러오기
        user_view_data =  pd.read_csv('./yorizori/user_view_recipe_log.csv').drop(['view_log_id','created_time'],axis=1)
        recipe_category_data = pd.read_csv('./yorizori/recipe_category.csv').drop(['category_id'],axis=1)
        user_info_data = pd.read_csv('./yorizori/yorizori_user.csv').drop(['created_time','updated_time','image_address','nickname','oauth_division'],axis=1)
        mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','level','time'],axis=1)

        user_rate_data.columns =['star_count','user_token_id','recipe_id']
        print(age1,"여자")
            
        
        df_user2,thisplus2,nb2,userv2=make_pivot_table(user_info_data,user_view_data,user_rate_data,recipe_category_data,'user_token_id','recipe_id','values',age1,"여자")
    
    #df_user_recipe_ratings , thisplus,nb
        if(df_user2.shape[0]<3):
            df_svd_pred = df_svd_pred_all
        else :
            df_svd_pred=filter_df(df_user2)
            filepath = './yorizori_predict/yorizori_predict_matrix'+age1+'_여자.csv'
            df_svd_pred.to_csv(filepath, index=False)
        thisplus2.to_csv('./yorizori_predict/yorizori_user_values'+age1+'_여자.csv',index=False)
        with open('./yorizori_predict/yorizori_user_index_info'+age1+'_여자.txt','w',encoding='UTF-8') as f:
            for name in nb2:
                f.write(name+'\n')    
# In[250]:
if __name__ =="__main__":

    

    #recommend_userid = 'abcde123'
    recommend_maker()

    
