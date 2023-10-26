
import pandas as pd #csv파일 읽어오는 모듈
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import time
from dateutil.relativedelta import relativedelta
import math
import warnings; warnings.simplefilter('ignore')

def filtering_today_recipe(md):
    now = datetime.now()

    lastyear_S = now - relativedelta(years=1) - relativedelta(days=10)
    lastyear_E = lastyear_S + relativedelta(days=20)
    
    thisyear = now - relativedelta(days=10)
    
    target_recipe = md[(md['created_time']>lastyear_S) & (md['created_time']<lastyear_E) | (md['created_time']>thisyear)]

    target_recipe['normal_view_count']=target_recipe['recipe_view_count'].apply(lambda x: 1 if x>2000 else (x-1)/1999)

    target_recipe['n_star_count'] = target_recipe['star_count'].apply(lambda x: 0.3*x if x<3 else 0.0875*(x-2)**2+0.2125)

    target_recipe['date_diff'] = target_recipe['created_time'].apply(lambda x: 0.2*((now-x).days%365) if thisyear.year==x.year else((now-x).days%365 if (now-x).days%365<50 else (x-now).days%365))

    target_recipe['n_review_count'] = target_recipe['review_count'].apply(lambda x: 1 if x>9 else (np.log(target_recipe['review_count']+1)))
    
    #(np.log(target_recipe['review_count']+2)*1.6)
    
    target_recipe['review_count']
    return target_recipe

def input_random_data(md):
    
    for i in range(len(md)):
        md['review_count'][i] = np.random.randint(0,200)
        md['star_count'][i] = round(np.random.uniform(0,5),1)
        
    return md


def recommend_today_recipe(recipe_df):
    table = filtering_today_recipe(recipe_df)
    table['total_score'] = (table['n_star_count']+table['n_review_count']+table['normal_view_count'])/(1+table['date_diff'])
    #나눠지는 값이 3부터여야 날짜에 의한 데이터의 변화가 심하지 않다고 생각함.
    table = table.sort_values('total_score', ascending = False)
    return table[:12]['recipe_id'].tolist(), table








switching = True #이거 False하고 하면 원본으로 실행될거에요!


if __name__ =="__main__":

    md = pd.read_csv('./yorizori/recipe_yorizori.csv').drop({'updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','user_token_id','level','time','version'},axis=1)
    md['created_time']= pd.to_datetime(md['created_time'])

    if(switching):
        md = input_random_data(md)
    
    target_recipe, table = recommend_today_recipe(md)
    
    print(target_recipe)
    #table
    