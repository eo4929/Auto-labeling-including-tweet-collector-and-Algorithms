
import tweepy
import base64
import requests
import json
import pandas as pd

from preprocessing_nlp import TextPreprocessing, ForClustering, VocaRepository, CS665_comparison

# 아래 있는 것들은 not 프리미엄 API
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

auto = tweepy.OAuthHandler(consumer_key, consumer_secret)
auto.set_access_token(access_token, access_token_secret)
twitter_api = tweepy.API(auto)

COUNT = 100

class Collectorbyteepy:
    def __init__(self):
        self.message_lst = []
    def search_by_keyword(self,keyword):
        #total_message = twitter_api.search(keyword, count=100)
        total_message = twitter_api.search(keyword, count=100, lang='en', geocode="-37.8136,144.963,10mi" )

        for tweet in total_message:
            self.message_lst.append(tweet.text)

        return self.message_lst

def main_bytweetpy():
    collector = Collectorbyteepy()
    res = collector.search_by_keyword("melbourne")
    print('length of total messages: ',len(res))
    print(res)

    res = collector.search_by_keyword("melbourne") # next로 넘어가는 거 어케하지 알아보기
    print('length of total messages: ', len(res))
    print(res)

    res = collector.search_by_keyword("melbourne")
    print('length of total messages: ', len(res))
    print(res)


import GetOldTweets3 as got
from bs4 import BeautifulSoup
#import datetime
from datetime import datetime, timedelta
import time
from random import uniform
from tqdm import tqdm_notebook,notebook

import time

class CollectorbyGetOldTweets3:
    def __init__(self):
        self.start = None
        self.end = None
        self.days_range = []
        self.tweet_lst = []

    def set_days_range(self,start,end):
        '''
        :param start: "2019-04-21"
        :param end: "2019-04-21"
        :return:
        '''
        self.start = datetime.strptime(start, "%Y-%m-%d")
        self.end = datetime.strptime(end, "%Y-%m-%d")
        date_generated = [self.start + timedelta(days=x) for x in range(0, (self.end - self.start).days)]

        for date in date_generated:
            self.days_range.append(date.strftime("%Y-%m-%d"))

        print("=== 설정된 트윗 수집 기간은 {} 에서 {} 까지 입니다 ===".format(self.days_range[0], self.days_range[-1]))
        print("=== 총 {}일 간의 데이터 수집 ===".format(len(self.days_range)))

    def collect_tweet(self,search_keyword=None,max_tweet=-1,lan='en',location='Melbourne, Australia',radius='5mi'):
        start_date = self.days_range[0]
        end_date = (datetime.strptime(self.days_range[-1], "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")  # setUntil이 끝을 포함하지 않으므로, day + 1
        #end_date = self.days_range[-1]

        # 트윗 수집 기준 정의
        tweetCriteria = got.manager.TweetCriteria().setNear(location) \
            .setWithin(radius) \
            .setSince(start_date) \
            .setUntil(end_date) \
            .setLang(lan) \
            .setMaxTweets(-1) # 나중에 -1로 바꾸기

        print("Collecting data start.. from {} to {}".format(self.days_range[0], self.days_range[-1]))
        start_time = time.time()

        tweet = got.manager.TweetManager.getTweets(tweetCriteria)

        print("Collecting data end.. {0:0.2f} Minutes".format((time.time() - start_time) / 60))
        print("=== Total num of tweets is {} ===".format(len(tweet)))

        #for index in tqdm_notebook(tweet):
        try:
            #for index in notebook.tqdm(tweet):
            for index in tqdm_notebook(tweet):
                username = index.username
                content = index.text
                tweet_date = index.date.strftime("%Y-%m-%d")
                tweet_time = index.date.strftime("%H:%M:%S")

                each_info_lst = [tweet_date, tweet_time, username, content]
                self.tweet_lst.append(each_info_lst)

                time.sleep(uniform(1, 2))
        except:
            twitter_df = pd.DataFrame(self.tweet_lst, columns=['date', 'time', 'user_name', 'content'])
            twitter_df.to_csv("sample_twitter_data_{}_to_{}.csv".format(self.days_range[0], self.days_range[-1]))
            print("except ouccur So, ")
            print("=== {} tweets are successfully saved ===".format(len(self.tweet_lst)))

        twitter_df = pd.DataFrame(self.tweet_lst, columns=['date','time','user_name','content'])
        twitter_df.to_csv("sample_twitter_data_{}_to_{}.csv".format(self.days_range[0], self.days_range[-1]))
        print("=== {} tweets are successfully saved ===".format(len(self.tweet_lst)))


class CollectorbyStandardAPI:
    def __init__(self): # API connection
        self.key_secret = '{}:{}'.format(consumer_key, consumer_secret).encode('ascii')
        self.b64_encoded_key = base64.b64encode(self.key_secret)
        self.b64_encoded_key = self.b64_encoded_key.decode('ascii')

        self.base_url = 'https://api.twitter.com/'
        self.auth_url = '{}oauth2/token'.format(self.base_url)

        self.auth_headers = {
            'Authorization': 'Basic {}'.format(self.b64_encoded_key),
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }
        self.auth_data = {
            'grant_type': 'client_credentials'
        }
        try:
            self.response = requests.post(self.auth_url, headers=self.auth_headers, data=self.auth_data)
        except:
            print('requests.post failed')

    def print_status_code(self):
        print('status_code:',self.response.status_code)

    def collect_tweet(self,query,count,until_date=None): # 한 request 사용
        '''
        :param query: ex) '우한폐렴 OR 코로나 OR ...'
        :param count: 10
        :param until_date: ex) 2015-07-19
        :return: -
        '''
        access_token = self.response.json()['access_token'] # Maximum number of tweets returned from a single token is 18,000
        search_headers = {
            'Authorization': 'Bearer {}'.format(access_token)
        }
        search_params = {
            'q': query,
            'result_type': 'mixed',  # 'recent' or 'popular' 로도 지정 가능
            #'geocode': "-37.817, 144.963, 5mi", 왜 이거 넣으면 안뽑힐까? -> 아마 트윗 남길때 사람들이 geo 정보를 잘 안 남겨서 그런걸 수 있을 것 같다
            'lang': 'en',
            'until': until_date, # 음.. standAPI는 범위 지정이 안되는데, 프리미엄 API는 가능한지 확인하기
            'count': count,  # 디폴트 값은 15이며, 최대 100까지 지정 가능
            'retryonratelimit': True  # rate limit에 도달했을 때 자동으로 다시 시도
        }
        search_url = '{}1.1/search/tweets.json'.format(self.base_url)

        try:
            search_response = requests.get(search_url,headers=search_headers,params=search_params)
        except:
            print('search_response failed')

        messages = json.loads(search_response.content)
        try:
            df = pd.DataFrame(messages['statuses'])
            #print(df.columns)
            major_df = df[['id_str', 'created_at', 'text', 'favorite_count', 'place', 'geo','coordinates' ,'user']] # 좀 더 재미있게 하려면 user ID 등 다른 것도 가져오자
        except:
            print('DataFrame is empty')
            #return None
            major_df = pd.DataFrame()
            return major_df

        return major_df

    def make_csv(self,df, name='test.csv'):
        df.to_csv(name)

    def collect_next_tweet(self,prev_df,query,count,until_date=None):
        #print(type(prev_df.iloc[-1]))
        if str( type(prev_df) ) == "<class 'NoneType'>":
            return pd.DataFrame()
        if prev_df.empty:
            return pd.DataFrame()
        last_row = prev_df.iloc[-1]
        #if last_row == None:
        #    return
        max_id = str( pd.to_numeric(last_row['id_str']) + 1 )

        access_token = self.response.json()['access_token']  # Maximum number of tweets returned from a single token is 18,000
        search_headers = {
            'Authorization': 'Bearer {}'.format(access_token)
        }
        search_params = {
            'q': query,
            'result_type': 'mixed',  # 'recent' or 'popular' 로도 지정 가능
            # 'geocode': "-37.817, 144.963, 5mi", 왜 이거 넣으면 안뽑힐까? -> 아마 트윗 남길때 사람들이 geo 정보를 잘 안 남겨서 그런걸 수 있을 것 같다
            'lang': 'en',
            'until': until_date,  # 음.. standAPI는 범위 지정이 안되는데, 프리미엄 API는 가능한지 확인하기
            'count': count,  # 디폴트 값은 15이며, 최대 100까지 지정 가능
            'max_id': max_id,
            'retryonratelimit': True  # rate limit에 도달했을 때 자동으로 다시 시도
        }
        search_url = '{}1.1/search/tweets.json'.format(self.base_url)

        try:
            search_response = requests.get(search_url,headers=search_headers,params=search_params)
        except:
            print('search_response in collect_next_tweet failed')
        #time.sleep(5.5)
        messages = json.loads(search_response.content)
        try:
            df = pd.DataFrame(messages['statuses'])
            #print(df.columns)
            major_df = df[['id_str', 'created_at', 'text', 'favorite_count', 'place', 'geo','coordinates' ,'user']] # 좀 더 재미있게 하려면 user ID 등 다른 것도 가져오자
        except:
            print('DataFrame in collect_next_tweet is empty')
            return None

        return major_df

    def make_originText_lst(self,df):
        lst =[]
        for idx in df.index:
            lst.append( df.loc[idx,'text'] )
        return lst

from datetime import datetime, timedelta

#def collectorEverydayByStandard(collector,keyword_list,until_date):


def main_standard():
    until_date = '2020-12-24'
    file_name = 'melbourne_until_201224.csv'
    keyword1 = 'melbourne'
    keyword2 = '(melbourne AND event)'
    keyword3 = '(melbourne AND traffic)'
    keyword4 = '(melbourne AND situation)'
    keyword5 = '(melbourne AND disaster)'
    keyword6 = '(melbourne AND pollution)'
    keyword7 = '(melbourne AND news)' # 이게 의외로 많네
    keyword8 = '(melbourne AND incident)'
    keyword9 = '(melbourne AND accident)'
    keyword10 = '(melbourne AND police)'
    keyword11 = '(melbourne AND festival)'
    keyword12 = '(melbourne AND report)'


    collector = CollectorbyStandardAPI()
    collector.print_status_code()
    # data = collector.collect_tweet('love OR hate',100,'2020-09-04') # 생각해보니 키워드 안넣고 그냥 period랑 location 정보만 넣고 모든 트윗 다 긁어오는 게 좋긴할텐데..
    # data = collector.collect_tweet('point_radius: [-37.8136 144.96 10mi]', 100) # mi 이 마일이니 킬로에서 마일로 바꾸기 --> 반경 8km 로 할꺼니 5마일 넣어라
    #data = collector.collect_tweet('place:ChIJ90260rVG1moRkM2MIXVWBAQ (date OR love)', 100)
    data = collector.collect_tweet(keyword1, 100, until_date) # 아 일주일 전부터 주어진 날짜 -1인 날 까지의 데이터를 수집하는거구나
    #collector.make_csv(data,'melbourne_and_event_201102.csv')
    print(data)

    # keyword1
    print('keyword1 step')
    next_data = collector.collect_next_tweet(data,keyword1,100,until_date)
    #print(next_data)
    data = data.append(next_data)

    for i in range(50):
        next_data = collector.collect_next_tweet(next_data, keyword1, 100, until_date)
        #print(next_data)
        if len(next_data) == 1 or next_data.empty:
            break
        data = data.append(next_data)

    # keyword2
    print('keyword2 step')
    data2 = collector.collect_tweet(keyword2, 100, until_date)
    data = data.append(data2)
    next_data2 = collector.collect_next_tweet(data2, keyword2, 100, until_date)
    # print(next_data)
    data = data.append(next_data2)

    for i in range(10):
        next_data2 = collector.collect_next_tweet(next_data2, keyword2, 100, until_date)
        # print(next_data)
        if len(next_data2) == 1 or next_data2.empty:
            break
        data = data.append(next_data2)

    # keyword3
    print('keyword3 step')
    data3 = collector.collect_tweet(keyword3, 100, until_date)
    data = data.append(data3)
    next_data3 = collector.collect_next_tweet(data3, keyword3, 100, until_date)
    # print(next_data)
    data = data.append(next_data3)

    for i in range(10):
        next_data3 = collector.collect_next_tweet(next_data3, keyword3, 100, until_date)
        # print(next_data)
        if len(next_data3) == 1 or next_data3.empty:
            break
        data = data.append(next_data3)

    # keyword4
    print('keyword4 step')
    data4 = collector.collect_tweet(keyword4, 100, until_date)
    data = data.append(data4)
    next_data4 = collector.collect_next_tweet(data4, keyword4, 100, until_date)
    # print(next_data)
    data = data.append(next_data4)

    for i in range(10):
        next_data4 = collector.collect_next_tweet(next_data4, keyword4, 100, until_date)
        # print(next_data)
        if len(next_data4) == 1 or next_data4.empty:
            break
        data = data.append(next_data4)

    # keyword5
    print('keyword5 step')
    data5 = collector.collect_tweet(keyword5, 100, until_date)
    data = data.append(data5)
    next_data5 = collector.collect_next_tweet(data5, keyword5, 100, until_date)
    # print(next_data)
    data = data.append(next_data5)

    for i in range(10):
        next_data5 = collector.collect_next_tweet(next_data5, keyword5, 100, until_date)
        # print(next_data)
        if len(next_data5) == 1 or next_data5.empty:
            break
        data = data.append(next_data5)

    print('keyword6 step')
    data6 = collector.collect_tweet(keyword6, 100, until_date)
    data = data.append(data6)
    next_data6 = collector.collect_next_tweet(data6, keyword6, 100, until_date)
    # print(next_data)
    data = data.append(next_data6)

    for i in range(10):
        next_data6 = collector.collect_next_tweet(next_data6, keyword6, 100, until_date)
        # print(next_data)
        if len(next_data6) == 1 or next_data6.empty:
            break
        data = data.append(next_data6)

    print('keyword7 step')
    data7 = collector.collect_tweet(keyword7, 100, until_date)
    data = data.append(data7)
    next_data7 = collector.collect_next_tweet(data7, keyword7, 100, until_date)
    # print(next_data)
    data = data.append(next_data7)

    for i in range(10):
        next_data7 = collector.collect_next_tweet(next_data7, keyword7, 100, until_date)
        # print(next_data)
        if len(next_data7) == 1 or next_data7.empty:
            break
        data = data.append(next_data7)

    print('keyword8 step')
    data8 = collector.collect_tweet(keyword8, 100, until_date)
    data = data.append(data8)
    next_data8 = collector.collect_next_tweet(data8, keyword8, 100, until_date)
    # print(next_data)
    data = data.append(next_data8)

    for i in range(10):
        next_data8 = collector.collect_next_tweet(next_data8, keyword8, 100, until_date)
        # print(next_data)
        if len(next_data8) == 1 or next_data8.empty:
            break
        data = data.append(next_data8)

    print('keyword9 step')
    data9 = collector.collect_tweet(keyword9, 100, until_date)
    data = data.append(data9)
    next_data9 = collector.collect_next_tweet(data9, keyword9, 100, until_date)
    # print(next_data)
    data = data.append(next_data9)

    for i in range(10):
        next_data9 = collector.collect_next_tweet(next_data9, keyword9, 100, until_date)
        # print(next_data)
        if len(next_data9) == 1 or next_data9.empty:
            break
        data = data.append(next_data9)

    print('keyword10 step')
    data10 = collector.collect_tweet(keyword10, 100, until_date)
    data = data.append(data10)
    next_data10 = collector.collect_next_tweet(data10, keyword10, 100, until_date)
    # print(next_data)
    data = data.append(next_data10)

    for i in range(10):
        next_data10 = collector.collect_next_tweet(next_data10, keyword10, 100, until_date)
        # print(next_data)
        if len(next_data10) == 1 or next_data10.empty:
            break
        data = data.append(next_data10)

    print('keyword11 step')
    data11 = collector.collect_tweet(keyword11, 100, until_date)
    data = data.append(data11)
    next_data11 = collector.collect_next_tweet(data11, keyword11, 100, until_date)
    # print(next_data)
    data = data.append(next_data11)

    for i in range(10):
        next_data11 = collector.collect_next_tweet(next_data11, keyword11, 100, until_date)
        # print(next_data)
        if len(next_data11) == 1 or next_data11.empty:
            break
        data = data.append(next_data11)

    print('keyword12 step')
    data12 = collector.collect_tweet(keyword12, 100, until_date)
    data = data.append(data12)
    next_data12 = collector.collect_next_tweet(data12, keyword12, 100, until_date)
    # print(next_data)
    data = data.append(next_data12)

    for i in range(10):
        next_data12 = collector.collect_next_tweet(next_data12, keyword12, 100, until_date)
        # print(next_data)
        if len(next_data12) == 1 or next_data12.empty:
            break
        data = data.append(next_data12)

    print('data = ')
    print(data)

    collector.make_csv(data,file_name)
    #lst = collector.make_originText_lst(data)
    # print(lst)
    # print('the number of messages=',len(lst))

preminum_key = ""
preminum_secret = ""

def get_utc_time(self, int_time):
    # datetime.utcfromtimestamp(int_time).strftime('%Y-%m-%d %H:%M:%S')
    # return (datetime.utcfromtimestamp(int_time) + timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S')
    return (datetime.utcfromtimestamp(int_time) + timedelta(hours=10)).strftime('%Y-%m-%d')

class CollectorbypremiumAPI:
    def __init__(self): # API connection
        self.key_secret = '{}:{}'.format(preminum_key, preminum_secret).encode('ascii')
        self.b64_encoded_key = base64.b64encode(self.key_secret)
        self.b64_encoded_key = self.b64_encoded_key.decode('ascii')

        self.base_url = 'https://api.twitter.com/1.1/tweets/search/'

        self.auth_url = '{}oauth2/token'.format(self.base_url)

        self.auth_headers = {
            'Authorization': 'Basic {}'.format(self.b64_encoded_key),
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }
        self.auth_data = {
            'grant_type': 'client_credentials'
        }
        try:
            self.response = requests.post(self.auth_url, headers=self.auth_headers, data=self.auth_data)
        except:
            print('requests.post failed')

    def collect_tweet(self, query, fromDate, toDate, maxResults=500): # Date는 UTC form이라고 함
        # 잠깐 .. query 할때 location 도 입력할 수 있는 지 확인해라
        '''
        :param query:
        '''
        search_url = '{}:product/:label.json'.format(self.base_url) # 여기서 :product이라는게 fullarive 넣으라는 말일수도?

        access_token = self.response.json()['access_token']
        search_headers = {
            'Authorization': 'Bearer {}'.format(access_token)
        }

        search_data = {
            'query': query,
            'fromDate': fromDate,
            'toDate': toDate,
            'maxResults': maxResults
        }
        try:
            search_response = requests.post(search_url, headers=search_headers, data=search_data) # 1분에 60번만 호출이 되니까 그렇게 호출되도로 코드 추가해넣기
        except:
            print('post on collect_tweet failed')
        try:
            json_data = search_response.json()
            print(json_data) # post로 얻어온 결과가 어떤 꼴인지 모르니 출력해보기
        except:
            print('jsonifing response failed')


from multiprocessing import Pool

def main():
    #collector = Collectorbyteepy()
    #res = collector.search_by_keyword("melbourne")
    #print(res)

    collector = CollectorbyStandardAPI()
    collector.print_status_code()
    #data = collector.collect_tweet('love OR hate',100,'2020-09-04') # 생각해보니 키워드 안넣고 그냥 period랑 location 정보만 넣고 모든 트윗 다 긁어오는 게 좋긴할텐데..
    #data = collector.collect_tweet('point_radius: [-37.8136 144.96 10mi]', 100) # mi 이 마일이니 킬로에서 마일로 바꾸기 --> 반경 8km 로 할꺼니 5마일 넣어라
    data = collector.collect_tweet('social OR data', 100)
    #print(data)
    #collector.make_csv(data)
    lst = collector.make_originText_lst(data)
    #print(lst)
    #print('the number of messages=',len(lst))

    text_preprocessor = TextPreprocessing()
    res = text_preprocessor.cleanMessage(lst)
    #print(res)

    res2 = text_preprocessor.nlpProcessing(res)
    #print(res2)

    #text_preprocessor.show_numof_eachword_and_total_words(res2)

    #res3 = text_preprocessor.create_word_to_vector_model(res2)
    #print(res3)
    #print(len(res3))

    #res4 = text_preprocessor.create_improved_word_to_vector_model(res2)
    #print(res4)
    #print(len(res4))

    test_message = [
        "Everyone in Melbourne is allowed to complain about lockdown and it doesn’t meant they are anti the lockdown measures it means we are experiencing a collective breakdown and want to scream. Okay.",
        "We are not allowed to travel more than 5km from home. We are only allowed out of our home for 1hr a day to shop and exercise. Masks must be worn at all times outside. No visitors allowed. These restrictions have just been extended for another 6 weeks. This is Melbourne Sept 2020.",
        "Under 50 cases in Melbourne today for the first time since June. If that rapid turnaround was from a vaccine or medicine it would need to be VERY effective. Lockdown medicine works. Costs are huge, will require years of rebuilding. But THOUSANDS OF LIVES saved. Kudos Victoria!",
        "A man labelled a risk to security by Australia's internal spy agency will be flown from coronavirus-crippled Melbourne to a detention centre in Western Australia to keep him safe from the pandemic.",
        "The insanity in Melbourne seems actually to be getting worse. Dan Andrews said it too: he is not following medical science (of course not!) but the modelling of computer science."
    ]

    res = text_preprocessor.cleanMessage(test_message)
    #print(res)
    res = text_preprocessor.nlpProcessing(res)

    text_preprocessor.show_numof_eachword_and_total_words(res)
    print(res)
    sentences = res
    res = text_preprocessor.create_improved_word_to_vector_model(res)
    #print(res)

    #text_preprocessor.visualize_data_with_PCA()

    cluster = ForClustering()
    cluster.set_word2vec_model( text_preprocessor.get_word2vec_model() )
    #print(res)
    #cluster.convert_dimension_lower_with_tsne()
    #cluster.visualize_data_with_tsne()
    #cluster.detect_and_remove_outliers()
    cluster.convert_dimension_lower_with_PCA()
    cluster.k_means()

# 아래가 oldtweet3으로 수집하는 코드
def main_tweet():
    tweet_collector = CollectorbyGetOldTweets3()
    tweet_collector.set_days_range("2019-12-07","2019-12-09") # 토탈 "2020-01-01","2020-01-11" 까지 모으기
    tweet_collector.collect_tweet() # 여기서 csv 까지 만듦

import numpy as np
from collections import Counter

def main_extracted_messages_information():
    sample_twitter_data_20191116_to_20191117 = pd.read_csv('./sample_twitter_data_2019-11-16_to_2019-11-17.csv')
    sample_twitter_data_20191123_to_20191124 = pd.read_csv('./sample_twitter_data_2019-11-23_to_2019-11-24.csv')
    sample_twitter_data_20191207_to_20191208 = pd.read_csv('./sample_twitter_data_2019-12-07_to_2019-12-08.csv')
    sample_twitter_data_20191218_to_20191220 = pd.read_csv('./sample_twitter_data_2019-12-18_to_2019-12-20.csv')
    sample_twitter_data_20200101_to_20200103 = pd.read_csv('./sample_twitter_data_2020-01-01_to_2020-01-03.csv')
    sample_twitter_data_20200104_to_20200106 = pd.read_csv('./sample_twitter_data_2020-01-04_to_2020-01-06.csv')
    sample_twitter_data_20200107_to_20200109 = pd.read_csv('./sample_twitter_data_2020-01-07_to_2020-01-09.csv')
    sample_twitter_data_20200125_to_20200126 = pd.read_csv('./sample_twitter_data_2020-01-25_to_2020-01-26.csv')
    sample_twitter_data_20200411_to_20200413 = pd.read_csv('./sample_twitter_data_2020-04-11_to_2020-04-13.csv')

    num_tweet = len(sample_twitter_data_20191116_to_20191117) + len(sample_twitter_data_20191123_to_20191124) + len(sample_twitter_data_20191207_to_20191208) + len(sample_twitter_data_20191218_to_20191220) + \
                len(sample_twitter_data_20200101_to_20200103) + len(sample_twitter_data_20200104_to_20200106) + len(sample_twitter_data_20200107_to_20200109) + len(sample_twitter_data_20200125_to_20200126) + len(sample_twitter_data_20200411_to_20200413)

    print('the num of different tweet=', num_tweet)

    lst_20191116_to_20191117 = list(np.array(sample_twitter_data_20191116_to_20191117['user_name'].tolist()))
    lst_20191123_to_20191124 = list(np.array(sample_twitter_data_20191123_to_20191124['user_name'].tolist()))
    lst_20191207_to_20191208 = list(np.array(sample_twitter_data_20191207_to_20191208['user_name'].tolist()))
    lst_20191218_to_20191220 = list(np.array(sample_twitter_data_20191218_to_20191220['user_name'].tolist()))
    lst_20200101_to_20200103 = list(np.array(sample_twitter_data_20200101_to_20200103['user_name'].tolist()))
    lst_20200104_to_20200106 = list(np.array(sample_twitter_data_20200104_to_20200106['user_name'].tolist()))
    lst_20200107_to_20200109 = list(np.array(sample_twitter_data_20200107_to_20200109['user_name'].tolist()))
    lst_20200125_to_20200126 = list(np.array(sample_twitter_data_20200125_to_20200126['user_name'].tolist()))
    lst_20200411_to_20200413 = list(np.array(sample_twitter_data_20200411_to_20200413['user_name'].tolist()))

    lst_user = lst_20191116_to_20191117 + lst_20191123_to_20191124 + lst_20191207_to_20191208 + lst_20191218_to_20191220 + lst_20200101_to_20200103 + lst_20200104_to_20200106 + lst_20200107_to_20200109 + lst_20200125_to_20200126 + lst_20200411_to_20200413
    res = Counter(lst_user)
    num_users = 0
    for _ in res:
        num_users +=1
    print('the num of different users=',num_users)

def make_tweet_lst(csv_from_to):
    pd_sample_twitter_data = pd.read_csv(csv_from_to)

    lst_tweet = list(np.array(pd_sample_twitter_data['content'].tolist()))

    return lst_tweet

def extract_tweets_of_a_day_from_csv_toMakeCsv(): # 2020-
    pd_tweet = pd.read_csv('./sample_twitter_data_2020-04-11_to_2020-04-13.csv')
    #print(pd_tweet['date'])
    #pd_tweet = pd.to_datetime(pd_tweet['date'])
    mask = ((pd_tweet['date'] == '2020-04-11') | (pd_tweet['date'] == '2020-04-12'))
    pd_res = pd_tweet[mask]

    pd_res.to_csv('./sample_twitter_data_2020-04-11_12.csv')


def main_extract_meaningful_keywords():
    lst_tweet = make_tweet_lst('./sample_twitter_data_2020-04-11_to_2020-04-13.csv')

    text_preprocessor = TextPreprocessing()
    cleaned_lst_tweet = text_preprocessor.cleanMessage(lst_tweet)
    preprocessed_lst_tweet = text_preprocessor.nlpProcessing(cleaned_lst_tweet)

    sorted_res = text_preprocessor.show_numof_eachword_and_total_words(preprocessed_lst_tweet)
    #text_preprocessor.create_improved_word_to_vector_model(preprocessed_lst_tweet)

    #cluster = ForClustering()
    #cluster.set_word2vec_model(text_preprocessor.get_word2vec_model())
    # print(res)
    # cluster.convert_dimension_lower_with_tsne()
    # cluster.visualize_data_with_tsne()
    # cluster.detect_and_remove_outliers()
    #cluster.convert_dimension_lower_with_PCA()
    #df_with_keywords = cluster.k_means() # k개의 centroid를 포함한 데이터 프레임을 반환

    #cluster.make_K_keyword_withFrequency(df_with_keywords,sorted_res,-1)


def main_voca():
    voca = VocaRepository()
    #voca.synonyms()
    #voca.hyponyms()
    #voca.extractKeywords()
    #voca.meronyms()
    #voca.test()
    res = voca.createVocaList()
    print(res)

def main_CS665():
    cs665 = CS665_comparison()
    cs665.read_tweets()
    #print(cs665.sample_twitter_data_20191116_to_20191117)
    lst = cs665.make_originText_lst(cs665.sample_twitter_data_20200411_to_20200413)

    text_preprocessor = TextPreprocessing()

    res = text_preprocessor.cleanMessage(lst)
    res = text_preprocessor.nlpProcessing(res)

    #text_preprocessor.show_numof_eachword_and_total_words(res)
    #print(res)
    sentences = res
    res = text_preprocessor.create_improved_word_to_vector_model(res)

    cluster = ForClustering()
    cluster.set_word2vec_model(text_preprocessor.get_word2vec_model())
    # cluster.convert_dimension_lower_with_tsne()
    cluster.convert_dimension_lower_with_PCA()

    voca = VocaRepository()
    vocaList = voca.createVocaList()

    cluster.make_weight_array(vocaList)

    caseList = [cs665.case1_event, cs665.case2_event, cs665.case2_sit, cs665.case4_sit]

    print('random: ')
    centers = cluster.CS665_kmeans("random")
    res = cluster.compute_acc(centers, caseList[3])
    print('res(random) = ', res)

    print('k-means++: ')
    centers = cluster.CS665_kmeans("k-means++")
    res = cluster.compute_acc(centers, caseList[3])
    print('res(k-means++) = ', res)

    print('weighted k-means++: ')
    centers = cluster.CS665_kmeans("weighted k-means++")
    res = cluster.compute_acc(centers, caseList[3])
    print('res(weighted k-means++) = ', res)


def main_test():
    test_message = [
        "Everyone in Melbourne is allowed to complain about lockdown and it doesn’t meant they are anti the lockdown measures it means we are experiencing a collective breakdown and want to scream. Okay.",
        "We are not allowed to travel more than 5km from home. We are only allowed out of our home for 1hr a day to shop and exercise. Masks must be worn at all times outside. No visitors allowed. These restrictions have just been extended for another 6 weeks. This is Melbourne Sept 2020.",
        "Under 50 cases in Melbourne today for the first time since June. If that rapid turnaround was from a vaccine or medicine it would need to be VERY effective. Lockdown medicine works. Costs are huge, will require years of rebuilding. But THOUSANDS OF LIVES saved. Kudos Victoria!",
        "A man labelled a risk to security by Australia's internal spy agency will be flown from coronavirus-crippled Melbourne to a detention centre in Western Australia to keep him safe from the pandemic.",
        "The insanity in Melbourne seems actually to be getting worse. Dan Andrews said it too: he is not following medical science (of course not!) but the modelling of computer science.",
        "hot hottest hotter conflict conflicts"
    ]

    nlp = TextPreprocessing()
    res = nlp.nlpProcessing(test_message)
    print(res)


if __name__ == '__main__':
    #main()
    #main_tweet()
    #main_bytweetpy()
    #main_voca() # 어휘집 테스트용
    #main_CS665()
    main_standard() # 이게 매일매일 메시지 수집기
    #main_test()
    #main_extracted_messages_information()
    #main_extract_meaningful_keywords()
    #extract_tweets_of_a_day_from_csv_toMakeCsv()
