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

import random
import warnings; warnings.simplefilter('ignore')

from datetime import datetime,timedelta
import time
from dateutil.relativedelta import relativedelta

from apscheduler.schedulers.background import BackgroundScheduler
import load_db_data
import yorizori_recommend_maker

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
# Flask 애플리케이션을 생성합니다.
app = Flask(__name__)
CORS(app)

# 오늘의 메뉴 원본 실행하기.(FALSE)
switching = False #이거 False하고 하면 원본으로 실행될거에요!


def save_to_data_import():
    # csv 데이터베이스에서 요청 후 적용하기
    try:
        load_db_data.load_db_data_csv()
        print("데이터 요청 완료")
        
        # predict_matrix.csv 생성하기
        yorizori_recommend_maker.recommend_maker()

        # user recommend init
        md = pd.read_csv('./yorizori/recipe_yorizori.csv')
        df_predict = pd.read_csv('./yorizori_predict/yorizori_predict_matrix.csv')
        user_info = pd.read_csv('./yorizori_predict/yorizori_user_values.csv')
        f = open('./yorizori_predict/yorizori_user_index_info.txt','r')
        df_user = f.readlines()

        # today recommend init
        md_today = pd.read_csv('./yorizori/recipe_yorizori.csv').drop({'updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','user_token_id','level','time','version'},axis=1)
        md_today['created_time']= pd.to_datetime(md_today['created_time'])
        print("1시간마다 요청")  # 보낸 후에 출력되는 메시지

    except:
        print("데이터 요청 실패")



scheduler = BackgroundScheduler()
scheduler.add_job(save_to_data_import, 'interval', minutes=5)
scheduler.start()



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
        

# 서술어 변환하기 ex : 한다 -> 하고
def predicate_conversion(response):
    변환_규칙1 = {
        '한다': '하고',
        '인다': '이고',
        '짠다': '짜고',
        '낸다': '내고',
        '썬다': '썰고',
        '뺀다': '빼고',
        '는다': '고',
        '준다': '주고',
        '힌다': '히고',
        '른다': '르고',
        '린다': '리고',
        '긴다': '기고',
        '둔다': '두고',
        '든다': '들고',
        '뺀다': '빼고',
        '진다': '지고',
        '하다': '하고',
        '친다': '치고',
        '킨다': '키고',
        '찐다': '찌고'
    }

    변환_규칙2 = {
        '한다': '한 뒤',
        '인다': '인 뒤',
        '짠다': '짠 뒤',
        '낸다': '낸 뒤',
        '썬다': '썬 뒤',
        '뺀다': '뺀 뒤',
        '는다': '은 뒤',
        '준다': '준 뒤',
        '힌다': '힌 뒤',
        '른다': '른 뒤',
        '린다': '린 뒤',
        '긴다': '긴 뒤',
        '둔다': '둔 뒤',
        '든다': '든 뒤',
        '뺀다': '뺀 뒤',
        '진다': '진 뒤',
        '하다': '한 뒤',
        '친다': '친 뒤',
        '킨다': '킨 뒤',
        '찐다': '찐 뒤'
    }

    변환_규칙3 = {
        '한다': '한 후',
        '인다': '인 후',
        '짠다': '짠 후',
        '낸다': '낸 후',
        '썬다': '썬 후',
        '뺀다': '뺀 후',
        '는다': '은 후',
        '준다': '준 후',
        '힌다': '힌 후',
        '른다': '른 후',
        '린다': '린 후',
        '긴다': '긴 후',
        '둔다': '둔 후',
        '든다': '든 후',
        '뺀다': '뺀 후',
        '진다': '진 후',
        '하다': '한 후',
        '친다': '친 후',
        '킨다': '킨 후',
        '찐다': '찐 후'
    }

    random_list = [변환_규칙1, 변환_규칙2, 변환_규칙3]

    변환_규칙 = random.choice(random_list)


    for 규칙, 변환 in 변환_규칙.items():
        if response[-3:].find(규칙) != -1: #마지막 3글자만 체크해서 변환하기
            return response[:-3] + response[-3:].replace(규칙, 변환 + ',') # 하고, 처럼 콤마붙여주기
    return response

def conjunctions_conversion(result_response):

    conj_list = [
        result_response + ". 그리고",
        result_response + ". 그 후",
        result_response + ". 그 뒤",
        result_response + ". 그런 뒤",
        result_response + ". 이후"
    ]

    rand_response = random.choice(conj_list)
    return rand_response


### 오늘의 메뉴
def filtering_today_recipe(md):
    now = datetime.now()

    lastyear_S = now - relativedelta(years=1) - relativedelta(days=10)
    lastyear_E = lastyear_S + relativedelta(days=20)
    
    thisyear = now - relativedelta(days=10)
    
    target_recipe = md[(md['created_time']>lastyear_S) & (md['created_time']<lastyear_E) | (md['created_time']>thisyear)]

    target_recipe['normal_view_count']=target_recipe['recipe_view_count'].apply(lambda x: 1 if x>2000 else (x-1)/1999)

    target_recipe['n_star_count'] = target_recipe['star_count'].apply(lambda x: 0.3*x if x<3 else 0.0875*(x-2)**2+0.2125)

    target_recipe['date_diff'] = target_recipe['created_time'].apply(lambda x: 0.2*((now-x).days%365) if thisyear.year==x.year else((now-x).days%365 if (now-x).days%365<50 else (x-now).days%365))

    # 표현식 버전 오류로 인해 아래껏으로 수정: 의미상으로는 동일
    # target_recipe['n_review_count'] = target_recipe['review_count'].apply(lambda x: 1 if x>9 else (np.log(target_recipe['review_count']+1)))
    target_recipe['n_review_count'] = target_recipe['review_count'].apply(lambda x: 1 if x > 9 else np.log(x + 1))

    
    #(np.log(target_recipe['review_count']+2)*1.6)
    
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




# 레시피 유저 맞춤 추천
@app.route('/recommend/<string:id>')
def get_recipe_recommend(id):

    recommend_userid = id
    for i in range(len(df_user)):
        df_user[i]=df_user[i].split("\n")[0]
    
    mds=md.drop(['created_time','updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','user_token_id','reference_recipe','level','time'],axis=1)
    already_rated, predictions = recommend_recipe(df_predict, recommend_userid, mds, user_info, 50)
    #예측행렬,예측하는유저id, 레시피테이블 데이터프레임, 유저별 로그 데이터프레임, 몇개추천할지
    return predictions['recipe_id'].to_list()

# 오늘의 요리 추천
@app.route('/recommendtoday')
def get_today_recommend():

    # if(switching):
    # md_today = input_random_data(md_today)
    print(md_today)
    
    target_recipe, table = recommend_today_recipe(md_today)
    # print(target_recipe)
    return target_recipe




@app.route('/template', methods=['POST'])
def get_string_by_templates():
    try:
        # JSON 데이터를 파싱하여 Python 데이터로 변환
        data = request.get_json()
        # templates = request.get_json()
        print(data)
        
        # templates 작업 수행
        responses = []  # 여러 템플릿에 대한 응답을 저장할 리스트

        for template in data['template']:
            # json으로 넘겨받는 항목들을 변수에 저장하고 list에 추가한다.
            template_list = []
            mask_count = 0

            # 기존 : condition, ingredient, size, time, tool, action 순서
            # 현재 : condition, tool, ingredient, size, time, action 순서
            condition = template.get("condition")
            # template_list.append(condition)
            tool = template.get("tool")
            conver_tool = tool.replace(' ', '')
            template_list.append(conver_tool)

            ingredient = template.get("ingredient")

            # 재료 슬라이싱   ex) 부추, 스프, 라면  => 부추와스프와라면
            # 재료의 원활한 변환을 위해 '와'로 바꾼다.
            conver_ingredient = ingredient.replace(' ', '')
            conver_ingredient = conver_ingredient.replace(',', '와')
            # template_list.append(ingredient)
            
            size = template.get("size")
            # template_list.append(size)

            # 재료와 크기 언어를 합쳐봄 (test)
            ing_size = conver_ingredient+size
            template_list.append(ing_size)

            time = template.get("time").replace(' ', '') # 공백 제거
            template_list.append(time)

            # 액션을 얻고 공백이 있다면 공백을 대체
            action = template.get("action")
            action = action.replace('.', '') # 온점 제거
            conver_action = ''
            if ' ' in action: # action에 공백이 있다면 
                conver_action = action.replace(' ', '')
            else:
                conver_action = action

            template_list.append(conver_action)

            # 비어있는 배열 ?=> ''을 없앤다.
            template_list = [item for item in template_list if item != '']

            print(template_list)
            temp_num = len(template_list)

            # Finetuning 작업 수행
            try:
                tmp=template_list
                for i in range(temp_num-1):
                    sen, mask_count = sentence(tmp,mask_count)
                    response=fill_mask(sen)
                    print(f'응답 {i} : {response}')
                    tmp=response.split(' ')
            except:
                print("what")

            # ingredient 공백을 다시 돌려 놓기
            if conver_ingredient in response:
                response = response.replace(conver_ingredient, ingredient)
            # action 공백을 다시 돌려 놓기
            if conver_action in response:
                response = response.replace(conver_action, action)
            # tool 공백을 다시 돌려 놓기
            if conver_tool in response:
                response = response.replace(conver_tool, tool)

            ### 여기에 컨디션 붙여주기 작업 추가, 컨디션이 비어있지 않다면 condition 앞에 추가하기.
            if condition != '':
                response = condition + " " + response

            print(response)
            
            # 재료의 원활한 변환을 위해 '와'로 바꾸었던 콤마들을 다시 돌려놓는다.
            response = response.replace('와', ', ').replace('  ', ' ')
            ingredient = ingredient.replace('와', ', ').replace('  ', ' ')
            try:
                # 재료, 크기를 분리해서 나중에 붙여주기.
                index = response.find(ingredient)
                print(f'ingredient :{ingredient}')
                print(index)
                if index != 0: # 재료가 있을 때
                    if size == '': # 크기가 없을 경우 붙여쓰기
                        response = response[:index + len(ingredient)] + response[index + len(ingredient):]
                    else: # 재료가 있고 크기가 있을 경우 띄어쓰기
                        response = response[:index + len(ingredient)] + " " + response[index + len(ingredient):]
                else:
                    print("안아줘요")
                responses.append(response)
            except:
                responses.append(ingredient)         


            # return String
            print(f'response : {response}')


        # Responses 안에 있는 내용물들의 마지막 서술어를 변환해주기
        result_response = ""
        자연스러움빈도 = 0
        for i, result in enumerate(responses):

            if len(responses) == 1: # 한개인 템플릿이 왔을 경우
                if not result.endswith('.'):
                    result_response = result + '.'
                else:
                    result_response = result
                break
            elif len(responses) > 1: # 여러개의 템플릿이 왔을 경우
                if i == 0: # 템플릿이 처음일 경우
                    # 문장이 너무 길다면 끊어버리고 그리고, 그 후, 그 뒤로 처리하기
                    if len(result)< 25:
                        result_response = predicate_conversion(result)
                        자연스러움빈도 += 1
                    else:
                        result_response = conjunctions_conversion(result)
                    print(f'처음:{result_response}')
                elif i != len(responses)-1: # 템플릿이 여러개일경우 끝에것이 아니라면
                    # 문장이 너무 길다면 끊어버리고 그리고, 그 후, 그 뒤로 처리하기
                    if len(result)< 25 and 자연스러움빈도 == 0:
                        result_response = result_response + " " + predicate_conversion(result)
                    else:
                        result_response = result_response + " " + conjunctions_conversion(result)
                    print(f'여러개:{result_response}')
                else: # 마지막 템플릿 일 경우
                    result_response = result_response + " " + result + "."
                    print(f'결론:{result_response}')


        # 만약 time 에 '시간' 이라는 말이 있다면 시간간을 시간으로 바꿔주기
        if time.find('시간') != -1:
            pass
        else:
            result_response = result_response.replace('시간간', '시간')



        print(result_response)
            

        # 결과를 JSON 형태로 응답
        response_data = {'response': result_response}
        response_json = json.dumps(response_data, ensure_ascii=False)
        return Response(response_json, content_type='application/json; charset=utf-8')    
        # return jsonify(response_data), 200
    except Exception as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 400




# In[250]:
if __name__ =="__main__":

    # user recommend init
    md = pd.read_csv('./yorizori/recipe_yorizori.csv')
    df_predict = pd.read_csv('./yorizori_predict/yorizori_predict_matrix.csv')
    user_info = pd.read_csv('./yorizori_predict/yorizori_user_values.csv')
    f = open('./yorizori_predict/yorizori_user_index_info.txt','r')
    df_user = f.readlines()

    # today recommend init
    md_today = pd.read_csv('./yorizori/recipe_yorizori.csv').drop({'updated_time','authorship','dish_name','recipe_intro','recipe_thumbnail','reference_recipe','user_token_id','level','time','version'},axis=1)
    md_today['created_time']= pd.to_datetime(md_today['created_time'])

    # finetuning Init *
    model2 = TFBertForMaskedLM.from_pretrained('./recipe_finetuning')
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    pip = FillMaskPipeline(model=model2, tokenizer=tokenizer)

    # 서버 실행
    app.run(use_reloader=False, host='0.0.0.0', port=5000)

    

# %%
