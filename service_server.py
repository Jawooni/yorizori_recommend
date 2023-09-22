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

import sklearn
from transformers import TFBertForMaskedLM, BertTokenizerFast, FillMaskPipeline
import tensorflow as tf


import warnings; warnings.simplefilter('ignore')

from flask import Flask, request, jsonify, Response
import json
# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)


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



def sentence(sen_list,mask_count): #리스트로 받은 템플릿 단어들을 모델에 넣기위해 string화 시키는 과정. 한 단어 뒤에 [MASK]를 집어넣어 MASK를 채울수있게한다.
    a=''
    b=mask_count
    size = len(sen_list)
    for i in range(size):
        if(mask_count==i):
            a=a+sen_list[i]+'[MASK] '
            b=mask_count+1
        else:
            a=a+sen_list[i]+' '

    return a,b

def fill_mask(sen): #중간에 [MASK]가 낀 문장을 받아서 MASK를 채우는 함수. 모든 조사나 어미는 ##@ 태그를 가지니까 그 MASK에 나온게 ##이 있을때까지 돌리기

    tmp_dict = pip(sen,top_k=20) #MASK를 채울 후보군을 20개까지 검사한다. 이 이후로 나오는건 솔직히 너무 이상한게 나올거라 의미없다봄

    for i in range(len(tmp_dict)):
        if('##' in tmp_dict[i]['token_str']):
            return tmp_dict[i]['sequence']


@app.route('/recommend/<string:id>')
def get_recipe_recommend(id):

    recommend_userid = id
    for i in range(len(df_user)):
        df_user[i]=df_user[i].split("\n")[0]
    
    mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','user_token_id','reference_recipe','level','time'],axis=1)
    already_rated, predictions = recommend_recipe(df_predict, recommend_userid, mds, user_info, 50)
    #예측행렬,예측하는유저id, 레시피테이블 데이터프레임, 유저별 로그 데이터프레임, 몇개추천할지
    return predictions['recipe_id'].to_list()

@app.route('/template', methods=['POST'])
def get_string_by_templates():
    try:
        # JSON 데이터를 파싱하여 Python 데이터로 변환
        templates = request.get_json()
        
        # templates 작업 수행
        mask_count=0

        # json으로 넘겨받는 항목들을 변수에 저장하고 list에 추가한다.
        template_list = []

        # 기존 : condition, ingredient, size, time, tool, action 순서
        # 현재 : condition, tool, ingredient, size, time, action 순서

        condition = templates[0].get("condition")
        template_list.append(condition)

        tool = templates[0].get("tool")
        template_list.append(tool)

        ingredient = templates[0].get("ingredient")
        # template_list.append(ingredient)
        
        size = templates[0].get("size")
        # template_list.append(size)

        # 재료와 크기 언어를 합쳐봄 (test)
        ing_size = ingredient+size
        template_list.append(ing_size)

        time = templates[0].get("time")
        template_list.append(time)

        action = templates[0].get("action")
        template_list.append(action)

        # 비어있는 배열 ?=> ''을 없앤다.
        template_list = [item for item in template_list if item != '']

        print(template_list)
        temp_num = len(template_list)

        # Finetuning 작업 수행
        tmp=template_list
        for i in range(temp_num-1):
            sen, mask_count = sentence(tmp,mask_count)
            response=fill_mask(sen)
            print(f'응답 {i} : {response}')
            tmp=response.split(' ')


        print(response)
        # 재료, 크기 분할
        index = response.find(ingredient)
        if index != -1:
            response = response[:index + len(ingredient)] + " " + response[index + len(ingredient):]
        else:
            print("안아줘요")


        # return String
        print(f'response : {response}')

        # 결과를 JSON 형태로 응답
        response_data = {'response': response}
        response_json = json.dumps(response_data, ensure_ascii=False)
        return Response(response_json, content_type='application/json; charset=utf-8')    
        # return jsonify(response_data), 200
    except Exception as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 400

# In[250]:
if __name__ =="__main__":


    # recommend init
    md = pd.read_csv('./yorizori/recipe_yorizori.csv')
    df_predict = pd.read_csv('./yorizori_predict_matrix.csv')
    user_info = pd.read_csv('./yorizori_user_values.csv')
    f = open('./yorizori_user_index_info.txt','r')
    df_user = f.readlines()


    # finetuning Init *
    model2 = TFBertForMaskedLM.from_pretrained('./recipe_finetuning')
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    pip = FillMaskPipeline(model=model2, tokenizer=tokenizer)


    # 서버 실행
    app.run(use_reloader=False, host='0.0.0.0', port=5000)

    

# %%
