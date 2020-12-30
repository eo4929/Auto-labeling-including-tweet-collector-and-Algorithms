
import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from yellowbrick.cluster import KElbowVisualizer

from gensim.models import Word2Vec
import matplotlib.pyplot as plt

import pandas as pd

from collections import Counter
from operator import itemgetter

class TextPreprocessing:
    def __init__(self):
        self.embedding_model = None
    def cleanMessage(self,lst):
        '''
        :param lst: - 
        :return: 일단 하나의 문장으로 만들고, 나중에 토큰화 하도록 하자
        '''
        new_lst = []

        for each_text in lst:
            text = re.sub('RT @[\w_]+: ', '', each_text) # 리트윗한 표시 제거
            text = re.sub('@[\w_]+', '', text) # 다른 사람 언급한 것도 제거
            text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ' ',text)  # http로 시작되는 url 제거
            text = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{2,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", ' ',text)  # http로 시작되지 않는 url 제거

            text = re.sub('[#]+[0-9a-zA-Z_]+', ' ', text) # 해시만 제거가 맞나? 나중에 확인하기
            text = re.sub('[&]+[a-z]+', ' ', text) # &lt, &gt 같은 표현들도 제거

            text = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', text) # Remove Special Characters -> 아 이걸써야 let's 를 let s 해주는 군.. 문제는 covid_19 도 covid 19 으로 바꾼다는 게 아쉽네

            text = text.replace('\n', ' ')
            split_text = ' '.join(text.split())

            new_lst.append(split_text)

        return new_lst

    def nlpProcessing(self,lst):
        new_lst = []

        for each_text in lst:
            words = word_tokenize(each_text) # 1) 토큰화 
            stop = stopwords.words('english')
            words = [token for token in words if token not in stop] # 2) 불용어 제거

            words = [word for word in words if len(word) >= 3] # 3) 의미가 없는 짧은 단어들 제거
            words = [word.lower() for word in words] # 4) 소문자화

            lmtzr = WordNetLemmatizer()
            words = [lmtzr.lemmatize(word) for word in words] # 5) 표제어 추출
            words = [lmtzr.lemmatize(word, 'v') for word in words] # 5-2) 동사들에 대해서 따로 표제어화 시켜줘야 함

            stemmer = PorterStemmer() # 6) 어간 추출
            words = [ stemmer.stem(word) for word in words ]

            new_lst.append(words)

        return new_lst

    def show_numof_eachword_and_total_words(self,lst):
        total_keywords_lst = []
        for each_lst in lst:
            total_keywords_lst += each_lst
        res = Counter(total_keywords_lst)
        sorted_rest = sorted(res.items(), key=itemgetter(1), reverse=True)
        print(sorted_rest)

        cnt=0
        num_keyword=0
        for element in sorted_rest:
            cnt += element[1]
            num_keyword+=1
        print('total number of words=',cnt) # 순수 단어 개수
        print('the number of keyword=', num_keyword) # 단어 당 카운트를 센 개수

        return sorted_rest

    def plot_2d_graph(self,words, xs, ys):
        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, marker='o')

        for i, word in enumerate(words):
            plt.annotate(word, xy=(xs[i], ys[i]))

        plt.show()

    def create_word_to_vector_model(self,lst): # 내 문장들로 학습시키는 모델
        self.embedding_model = Word2Vec(lst,size=300,workers=4)
        word_vector = self.embedding_model.wv

        keys = word_vector.vocab.keys()
        #print(keys)
        new_lst = [ word_vector[k] for k in keys ]

        # 아래코드는 2차원에서 출력해보기 위함
        pca = PCA(n_components=2)
        xys = pca.fit_transform(new_lst)
        xs = xys[:,0]
        ys = xys[:,1]
        #self.plot_2d_graph(keys,xs,ys)

        return new_lst

    def get_word2vec_model(self):
        return self.embedding_model

    def create_improved_word_to_vector_model(self,lst): # 내 문장에서 추출한 키워드랑 구글 뉴스로 학습된 모델을 같이 병합해서 학습한 모델 -> 이렇게 해야 단어 벡터들 사이의 유사도가 더 정확한걸로 알려져있음
        '''
        :param lst: 
        :return: 임베딩된 벡터 리스트 반환
        '''
        
        self.embedding_model = Word2Vec(lst, size=300, workers=4, min_count=2) # min_count(단어에 대한 개수를 의미하는 듯.. 즉 default는 5개 이하의 단어에 대해선 무시하는구나..) 에 대한 threshold를 줄이니까 에러 해결..
                                                                          # 근데 실제 프리미엄 API로 수집한 메시지에서 단어 개수가 너무 많으면 min_count를 높이자 -> 그러면 빈도수가 낮은 단어들은 벡터 임베딩이 안되겠지만 모델 크기를 줄이려면 그래야 할 수도 있음
        file_name = 'GoogleNews-vectors-negative300.bin.gz'
        self.embedding_model.intersect_word2vec_format(fname=file_name,binary=True)

        word_vector = self.embedding_model.wv

        keys = word_vector.vocab.keys()
        #print(keys)
        new_lst = [word_vector[k] for k in keys]

        # 아래코드는 2차원에서 출력해보기 위함

        #self.plot_2d_graph(keys,xs,ys)
        self.embedding_model.save('word2vec.model')
        #word2vec_model = Word2Vec.load('word2vec.model')

        return new_lst

    def visualize_data_with_PCA(self):
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        # print(keys)
        new_lst = [word_vector[k] for k in keys]

        # 아래코드는 2차원에서 출력해보기 위함
        pca = PCA(n_components=2)
        xys = pca.fit_transform(new_lst)
        xs = xys[:, 0]
        ys = xys[:, 1]
        #self.plot_2d_graph(keys, xs, ys)


from nltk.corpus import wordnet as wn
class VocaRepository: # here
    def __init__(self):
        self.nounList = ['event','traffic','situation','disaster','pollution','news','incident','accident','police','festival','report','public','city','construction','citizen', 'civil', 'announce']
        #verbList = ['change'] # 이건 찾아봐야 할 듯 -> 교수님께 질문 드리기

        self.vocaList = [] # createVocaList 호출로 만들어진 어휘 저장소

        self.tab = '\t'

    def createVocaList(self):
        synsetsLst = self.hyponyms(self.nounList)
        self.extractKeywords(synsetsLst)
        #print( self.vocaList )
        #print( len(self.vocaList) )

        newVocaList = []
        def replace_():
            for each_voca in self.vocaList:
                newVocaList.append(each_voca.replace("_"," "))

        replace_()
        self.vocaList = newVocaList
        return self.vocaList

    def printSynonyms(self):
        '''
        :return: 동의어 리스트
        '''
        total_lst = []

        for synset in wn.synsets('event'): # 아 한 synset 은 해당 어휘에 대한 '한' 오브젝를 의미하는구나 -> 정의 , 동의어 등 포함
            print("{}: {}".format(synset.name(), synset.definition()))

            synonyms = ", ".join([lem.name() for lem in synset.lemmas()]) # 동의어 -> 자기 자신은 1개만 남기기
            print(self.tab + "synonyms: {}".format(synonyms))

            total_lst.append( synset.lemma_names() )

        print(total_lst)
        return total_lst

    def extractKeywords(self,synsetLst): # here 잠깐 일단 하위어랑 부분어 depth를 레벨5정도로 줄였음 너무 많아서
        '''
        synsetLst: 하위어 함수가 내뱉은 거
        :return:동의어와 자기 자신 단어 다 추출하기
        '''
        for synset in synsetLst:
            #print("{}: {}".format(synset.name(), synset.definition()))
            #synonymsLst = [st.lemma_names()[0] for st in synset]
            synonymsLst = synset.lemma_names()
            #print(synonymsLst)
            self.vocaList += synonymsLst



    # def hyponyms(self): # 하위어 구현
    #     total_lst = []
    #
    #     for synset in wn.synsets('event'):
    #         print( synset.hyponyms() )

    def DeepPrintHyponyms(self,synset):

        r_paths = []
        def hyponym_paths_helper(input_syns, paths, cnt):
            cnt +=1
            if cnt == 5:
                return
            if len(input_syns.hyponyms()) == 0:
                for path in paths:
                    r_paths.append(path)
            else:
                for hyponym in input_syns.hyponyms():
                    hyponym_paths_helper(hyponym, [path + [hyponym] for path in paths], cnt)

        hyponym_paths_helper(synset, [[]], 0)
        return r_paths

    def hyponyms(self, nounLst=None): # 이걸로 하위어들 모으고, 하위어당 동의어를 검색하는 식으로 하자
        hyponymList = []
        for noun in nounLst:
            for synset in wn.synsets(noun):
                lst_of_lst = self.DeepPrintHyponyms(synset)
                for Lst in lst_of_lst:
                    hyponymList = hyponymList + Lst
                hyponymSet = set(hyponymList)
                hyponymList = list(hyponymSet)
        #print(hyponymList)
        #print( len(hyponymList) )
        return hyponymList

    # def DeepPrintMeronyms(self,synset):
    #
    #     r_paths = []
    #     def meronyms_paths_helper(input_syns, paths):
    #         if len(input_syns.part_meronyms()) == 0:
    #             for path in paths:
    #                 r_paths.append(path)
    #                 #r_paths = r_paths + path
    #         else:
    #             for meronym in input_syns.part_meronyms():
    #                 meronyms_paths_helper(meronym, [path + [meronym] for path in paths])
    #
    #     meronyms_paths_helper(synset, [[]])
    #     return r_paths
    #
    # def meronyms(self, nounLst=None): # 이벤트 등 어휘에 대해서는 대부분 부분어가 없구나
    #     meronymList = []
    #     for synset in wn.synsets('disaster'):
    #         lst_of_lst = self.DeepPrintMeronyms(synset)
    #         for Lst in lst_of_lst:
    #             meronymList = meronymList + Lst
    #     meronymSet = set(meronymList)
    #     meronymList = list(meronymSet)
    #
    #     return meronymList

    def test(self):
        self.DeepPrintMeronyms(wn.synset('event.n.01'))

from sklearn.manifold import TSNE



class ForClustering:
    def __init__(self):
        self.embedding_model = None
        self.res_tsne = None

        self.new_lst = None # 300 차원의 임베딩 벡터
        self.reduced_new_lst = None  # 50 차원의 임베딩 벡터
        self.df_new_lst_without_outliers = None # 생각해보니 array로 관리할 필요가 없네.. 아웃라이어 제거부터는 df로 관리하자

        self.weight_array = None

    def set_word2vec_model(self,embedding_model):
        self.embedding_model = embedding_model

        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        self.new_lst = [word_vector[k] for k in keys]

    def plot_2d_graph(self,words, xs, ys):
        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, marker='o')

        for i, word in enumerate(words):
            plt.annotate(word, xy=(xs[i], ys[i]))

        #plt.show()

    def detect_and_remove_outliers(self): # 일단 실제 구현상에선 배제해보자..
        pd_new_lst = pd.DataFrame(self.new_lst)
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()

        pd_new_lst.index = [ k for k in keys ]
        #print(pd_new_lst)

        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        pred_outliers = clf.fit_predict(pd_new_lst)

        out = pd.DataFrame(pred_outliers)
        out.index = [ k for k in keys ]
        out = out.rename(columns={0: "out"})
        df_new_lst_with_outliers = pd.concat([pd_new_lst, out], 1)
        df_new_lst_with_outliers = df_new_lst_with_outliers.loc[ df_new_lst_with_outliers.out == 1 ]
        del df_new_lst_with_outliers['out']

        #print(df_new_lst_with_outliers)
        #print(df_new_lst_with_outliers)
        self.df_new_lst_without_outliers = df_new_lst_with_outliers
        #self.df_new_lst_without_outliers = df_new_lst_with_outliers.values
        self.visualize_data_with_PCA()


    def convert_dimension_lower_with_PCA(self): # 잠깐 근데 PCA를 적용하기 전에 data normalization을 통해 데이터의 평균이 0, 표준 편차가 1이 되도록 해야 하나..? 아직 모르겠다
        '''
        :return: pandas 형태로 뿌려보자
        '''

        #print( pd.DataFrame(new_lst, [k for k in keys]) )

        pca = PCA(n_components=50)
        self.reduced_new_lst = pca.fit_transform(self.new_lst)
        #print(self.reduced_new_lst)
        #self.visualize_data_with_PCA()
        return self.reduced_new_lst

    def make_weight_array(self,vocaList=None):
        '''
        :param vocaLst: cs665.vocaList 를 여기에 넣자
        :return: weigh_array 만들어지게 하고 이걸 CS665_kmeans 의 마지막 파라미터로 들어가게 하자
        '''
        # here
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        # print(keys)
        keyword_lst = [word_vector[k] for k in keys]

        keysList=list(keys) # 이게 클러스터링될 키워드 리스트 (no vector)

        self.weight_array = [0 for _ in range(len(keysList))]

        for i, eachKeyword in enumerate(keysList):
            for eachVoca in vocaList:
                try:
                    similarity = word_vector.similarity(eachKeyword,eachVoca)
                    if(similarity > 0.7): # 나중에는 유사도 높은애한테 더 높은 가중치를 주는 모델 고려할 순 있을 것임 지금은 귀찮..
                        self.weight_array[i] +=10 # 이걸 컨트롤하면 weighted k-means가 더 나알질수도 있음
                except: # eachVoca가 word2vec 모델에 없으면 인지를 못함.. 개들은 건너뛰자
                    continue
        return self.weight_array
        #print(self.weight_array)



    def CS665_kmeans(self,type="k-means++",weigh_array=None): # 일단 최종 결과 어떻게 뽑는지 먼저 보자
        '''
        :param type:  1) 'random', 2) "k-means++", 3) weighted k-means++
        :return:
        '''
        model = KMeans()
        visualizer = KElbowVisualizer(model, metric='calinski_harabasz', k=(3, 100))

        visualizer.fit(self.reduced_new_lst)
        # visualizer.show()
        K = visualizer.elbow_value_

        if K == None:
            K = 50
        print('K= ', K)

        # 파라미터로 model 선택
        model = None
        if type == "k-means++":
            model = KMeans(init="k-means++", n_clusters=K, random_state=0)
            model.fit(self.reduced_new_lst)
        elif type == "random":
            model = KMeans(init="random", n_clusters=K, random_state=0)
            model.fit(self.reduced_new_lst)
        elif type == "weighted k-means++":
            model = KMeans(init="k-means++", n_clusters=K, random_state=0)
            model.fit(self.reduced_new_lst , sample_weight= self.weight_array)

        #model.fit(self.reduced_new_lst)
        #xys = model.fit_transform(self.reduced_new_lst)
        #y_kmeans = model.predict(self.reduced_new_lst)
        # print(xys)

        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()

        #xs = xys[:, 0]
        #ys = xys[:, 1]
        # self.plot_2d_graph(keys, xs, ys)

        # 아래는 dataframe으로 뿌리기 위한 용도이구나
        pd_reduced_new_lst = pd.DataFrame(self.reduced_new_lst)
        keys = [k for k in keys]
        pd_keys = pd.DataFrame(keys)
        pd_keys = pd_keys.rename(columns={0: "keyword"})
        df = pd.concat([pd_reduced_new_lst, pd_keys], 1)
        # print(df)

        #plt.figure()
        #plt.scatter(xs, ys, c=y_kmeans, s=50, cmap='viridis')
        #words = df['keyword']
        #for i, word in enumerate(words):
        #    plt.annotate(word, xy=(xs[i], ys[i]))

        centers = model.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        pd_centers = pd.DataFrame(centers)
        # print(pd_centers)

        new = pd.concat([pd_centers, pd_keys], axis=1, join='inner')
        #print(new)
        # plt.show()

        return new

    def compute_acc(self, centers, eachCaseList): #d
        acc = 0
        centroids = centers['keyword']
        centroids = centroids.tolist()

        print('eachCaseList = ',eachCaseList)
        print('centroids = ', centroids)

        for centroid in centroids:
            if centroid in eachCaseList:
                acc += 1 # 10% 씩 증가
        return (acc / 12) * 100

    def k_means(self): # k개의 centroid를 반환
        model = KMeans()
        visualizer = KElbowVisualizer(model, metric='calinski_harabasz', k=(3, 100))

        visualizer.fit( self.reduced_new_lst )
        #visualizer.show()
        K = visualizer.elbow_value_

        if K == None:
            K = 50
        print('K= ',K)

        model = KMeans(init="k-means++", n_clusters=K, random_state=0)
        xys = model.fit_transform(self.reduced_new_lst)
        y_kmeans = model.predict(self.reduced_new_lst)
        #print(xys)

        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()

        xs = xys[:, 0]
        ys = xys[:, 1]
        #self.plot_2d_graph(keys, xs, ys)

        # 아래는 dataframe으로 뿌리기 위한 용도이구나
        pd_reduced_new_lst = pd.DataFrame(self.reduced_new_lst)
        keys = [k for k in keys]
        pd_keys = pd.DataFrame(keys)
        pd_keys = pd_keys.rename(columns={0: "keyword"})
        df = pd.concat([pd_reduced_new_lst, pd_keys], 1)
        #print(df)

        plt.figure()
        plt.scatter(xs, ys, c=y_kmeans, s=50, cmap='viridis')
        words = df['keyword']
        for i, word in enumerate(words):
            plt.annotate(word, xy=(xs[i], ys[i]))

        centers = model.cluster_centers_
        #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        pd_centers = pd.DataFrame(centers)
        #print(pd_centers)

        new = pd.concat([pd_centers, pd_keys], axis=1, join='inner')
        print(new)
        #plt.show()

        return new

    # 여기에 추출된 키워드 k개에 대한 frequency 도 카운팅하는 함수 만들자
    def make_K_keyword_withFrequency(self,df_keywords,sorted_res,num_rank):
        '''
        :param df_keywords:
        : sorted_res: 여기에 Counter 오브젝 들어옴
        :param num_rank: 상위 몇개까지의 키워드를 얻고 싶은지 넣기
        :return:
        '''
        lst_K_keyword_withFrequency = []
        for each_ele in sorted_res:
            mask = (df_keywords['keyword'] == each_ele[0])
            res = df_keywords[mask]
            if res.empty:
                continue
            else:
                lst_K_keyword_withFrequency.append(each_ele)

        print(lst_K_keyword_withFrequency)
        return lst_K_keyword_withFrequency[0:num_rank] # 근데 생각해보니 num_rank보다 적은 수의 단어가 있으면 에러 뜨겠구나

    def convert_dimension_lower_with_tsne(self):
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        # print(keys)
        new_lst = [word_vector[k] for k in keys]

        tsne_analyzer = TSNE(n_components=2) # 시각화를 위해 2차원으로 둠
        self.res_tsne = tsne_analyzer.fit_transform(new_lst)

    def visualize_data_with_tsne(self):
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        # print(keys)
        new_lst = [word_vector[k] for k in keys]

        # 아래코드는 2차원에서 출력해보기 위함
        xys = self.res_tsne
        xs = xys[:, 0]
        ys = xys[:, 1]
        self.plot_2d_graph(keys,xs,ys)

    def visualize_data_with_PCA(self):
        word_vector = self.embedding_model.wv
        keys = word_vector.vocab.keys()
        # print(keys)
        new_lst = [word_vector[k] for k in keys]

        #words = self.df_new_lst_without_outliers.index.tolist()

        pca = PCA(n_components=2)
        xys = pca.fit_transform(new_lst)
        xs = xys[:, 0]
        ys = xys[:, 1]
        self.plot_2d_graph(keys, xs, ys)




class CS665_comparison:
    def __init__(self):
        # 보카 어휘집
        self.voca = VocaRepository()
        self.vocaList = self.voca.createVocaList()

        # 불규칙 패턴에 해당하는 트윗 메시지들
        self.sample_twitter_data_20191116_to_20191117 = None
        self.sample_twitter_data_20191207_to_20191208 = None
        self.sample_twitter_data_20200411_to_20200413 = None



        # mapping을 위해 소문자로 변형했다고 말하기 + 단수 복수는 단수로 통일했다고 하기 + 얘네도 stemming 등 NLP 처리했다고 하기
        self.case1_event = ['artist', 'audience', 'song', 'hear', 'rock', 'pop', 'musician', 'music', 'band', 'melbourn', 'event', 'festival']
        self.case2_event = ['christmas', 'market', 'gift', 'christ', 'give', 'love', 'folk', 'anniversary', 'design', 'song', 'public', 'report']
        self.case2_sit = ['firefight', 'fire', 'bush', 'bushfire', 'wildfire','dockland','melbourn','southbound','alert','firefighters', 'wild','news']
        self.case4_sit = ['easter','happy','food','egg', 'shop','thank','christ', 'card', 'covid','corona','festival','song']


        self.read_tweets()

    def read_tweets(self):
        self.sample_twitter_data_20191116_to_20191117 = pd.read_csv('./SAC/sample_twitter_data_2019-11-16_to_2019-11-17.csv')
        self.sample_twitter_data_20191207_to_20191208 = pd.read_csv('./SAC/sample_twitter_data_2019-12-07_to_2019-12-08.csv')
        self.sample_twitter_data_20200411_to_20200413 = pd.read_csv('./SAC/sample_twitter_data_2020-04-11_to_2020-04-13.csv')

    def make_originText_lst(self,df):
        lst =[]
        for idx in df.index:
            lst.append( df.loc[idx,'content'] )
        return lst

