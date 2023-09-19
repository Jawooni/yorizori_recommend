#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
from transformers import TFBertForMaskedLM, BertTokenizerFast, FillMaskPipeline
import tensorflow as tf




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


# In[62]:

if __name__ =="__main__":


  # Init *
  model2 = TFBertForMaskedLM.from_pretrained('./recipe_finetuning')
  tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
  pip = FillMaskPipeline(model=model2, tokenizer=tokenizer)
  mask_count=0

  # json으로 넘겨받는 list 부분
  template_list = ['사과','당근','1cm','채썬다'] #이부분을 입력으로 받으시면 됩니다
  temp_num = len(template_list)

  tmp=template_list
  for i in range(temp_num-1):

    sen, mask_count = sentence(tmp,mask_count)
    a=fill_mask(sen)
    tmp=a.split(' ')

  # return String
  print(a)



