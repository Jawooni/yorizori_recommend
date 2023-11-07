import pandas as pd
from sqlalchemy import create_engine

# Database connection


def load_db_data_csv():
    username = 'admin'
    password = 'yorizori'
    host = 'yorizori-mysql.cbqb9r9uwmbw.ap-northeast-2.rds.amazonaws.com'
    port = 3306 # port number
    database = 'yorizori'

    connection_info = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'

    engine = create_engine(connection_info)
    conn = engine.connect()


    user_info= pd.read_sql_table('user', conn)
    user_comment = pd.read_sql_table('user_comment',conn)
    user_view_log = pd.read_sql_table('user_view_log',conn)
    recipe_category= pd.read_sql_table('recipe_category_tag', conn)
    recipe_info= pd.read_sql_table('recipe', conn)


    recipe_category.to_csv('./yorizori/recipe_category.csv',index=False)
    user_info.to_csv('./yorizori/yorizori_user.csv',index=False)
    user_comment.to_csv('./yorizori/user_comment.csv',index=False)
    recipe_info.to_csv('./yorizori/recipe_yorizori.csv',index=False)
    user_view_log.to_csv('./yorizori/user_view_recipe_log.csv',index=False)



if __name__ =="__main__":

    load_db_data_csv()