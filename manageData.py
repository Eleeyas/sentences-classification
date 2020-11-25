import numpy as np
import pandas as pd
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords

def replace(arr,number):
    items = np.array(arr)
    return np.where(items>1, number, items)

def manageData(text):
    stop_words = set(thai_stopwords())
    word_tokens = word_tokenize(text,engine='newmm')
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    text_join = " ".join(filtered_sentence)
    senten = text_join.split()
    dataList = pd.read_csv('dataTest.csv',usecols = lambda column : column not in ['class'])
    header = dataList.columns.values
    count_all = []
    for w in header:
        count_all.append(senten.count(w))
    x_train =  replace(dataList.values,1)
    y_train = pd.read_csv('dataTest.csv',usecols = ['class']).values
    x_test = np.column_stack(replace(count_all,1))
    return x_train, y_train, x_test
# ประกาศ class ที่ชื่อ manageData โดยค่าที่รับจะเป็นในรูปแบบ string เท่านั้น
# บรรทัดที่ 11-12 import thai_stopwords และ word_tokenize เพื่อเอามาตัดคำซึ่งในproject นี้จะใช้ระบบตัดคำเป็นแบบ newmm (Maximum Matching algorithm) ในการตัดคำภาษาไทย word_tokens คือ จะเก็บคำที่เอาออกจจากโยคแล้วไม่มีผลต่อประโยค 
# บรรทัดที่ 13-15 filtered_sentence output ที่ได้เป็นคำที่ผสมค่าempty จึ่งทำการ cleat ค่า empty ก่อนเนื่องจากค่า empty อาจะทำให้ถูกคิดตอน predict ด้วย
# บรรทัดที่ 16-17 import file data set ที่จะทำการ predict โดยจะแบ่ง data กับ header ออกจากกันเพื่อไม่ให้ header ถูก predict ด้วย
# บรรทัดที่ 18-20 ทำการนับคำที่เจอใน array โดยค่าoutput ที่ได้เป็น array number 2มิติ
# บรรทัดที่ 21 เมื่อได้ค่านับแล้ว convert ให้มีค่าแค่ 0 กับ 1
# บรรทัดที่ 22 เอาค่า target (nagative or positive) ใน file
# บรรทัดที่ 23 เอาค่า data set เพื่อเอามาทดสอบ classify
# บรรทัดที่ 24 จะ return data เพื่อเอาไป predict ต่อไปโดย x_train คือ เป็น data set ที่เอาไว้ train , y_train คือ target จะได้ออกมา( nagative or positive), x_test คือ data set จะทำการclassify โดยจะอ้างอิงจาก x_train,y_train
