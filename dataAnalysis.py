import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def regressionPrice():
    #todo 집값 예측 모듈 만들어야돼
    df = pd.read_csv('/Users/AhnJeongHyeon/Documents/GitHub/RealEstate/Data/APT19.06~20.0530.csv')
    df.columns = ['location', 'fnum', 'num', 'semiNum', 'name', 'size', 'ym', 'd', 'price', 'floor', 'buildY',
                  'address']
    #평 평당 가격 추가
    #공백 제거
    #df['price'].str.rstrip()
    #df['price'].str.lstrip()

    #***자료형 변형
    df['pyung'] = df['size'] / (3.3)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    #df['price'] = df['price'].astype(float)
    df['p_price'] = df['price']/df['pyung']
    df['result'] = 0

    #groupby로 집합 개수를 알 수 있는 함수 value_counts
    #print(df['location'].value_counts())

    #TODO regression
    '''
         logistic
       1. 강남이 강북보다 많이 오를 것이다
       2. 30평대가 더 많이 오를것이다
       3. BuildY가 높은 숫자가 많이 오를 것이다
       (4). 집값이 높으면 더 많이 오를 것이다  
       결 입력한 값 이상 오를 것인지 OX확률
       groupby max-min >= input 1 else 0
       1은 gangNam,gangBook을 df['location'] split하여 확인 1 else 0
       2는 pyung 30>=이면 1 else 0
       
       
       linear regression
       1. 강남이 강북보다 많이 오를 것이다
       2. 30평대가 더 많이 오를것이다
       3. BuildY가 높은 숫자가 많이 오를 것이다
       (4). 집값이 높으면 더 많이 오를 것이다  
       결과) 현재 집값 입력해서 얼마나 오를지 예측
    '''
    gangNam = ['강서구', '양천구', '구로구', '금천구', '영등포구', '관악구', '동작구', '서초구', '강남구', '송파구', '강동구']
    gangBook = ['마포구', '은평구', '서대문구', '종로구', '중구', '용산구', '강북구', '노원구', '성북구', '도봉구', '동대문구', '성동구', '광진구', '중랑구']
    #location, name, size로 group by하는데 max
    #todo 원하는것만 가져올 때 df[['']]
    temp = df[['location','name','pyung','buildY','result']]

    value = input("값을 입력하세요")
    for index,row in df.iterrows():
        print(df.iloc[index,'location'])
        if df.iloc[index,'location'] == df.iloc[index+1,'location']:
            df.iloc[index,'result'] = 1
            print(row)


def titanic():
    # http://hleecaster.com/ml-logistic-regression-example/
    passengers = pd.read_csv("titanic.csv")
    print(passengers.shape)#개수 찍어보기
    print(passengers.head())
    print(passengers.columns)
    #성별 나이 Pclass에 따라 생존율이 달랐을 것이다.
    #강남이냐 강북이냐로 나눌 수 있을듯
    #여자는1 남자는0
    passengers['Sex'] = passengers['Sex'].map({'female':1,'male':0})
    #age가 비어있는 경우 중간값으로

    passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
    #first는1 second=2 나머지0

    features = passengers[['Sex','Age','FirstClass','SecondClass', 'Parents/Children Aboard']]
    survival = passengers['Survived']

    train_features, test_features, train_labels, test_labels = train_test_split(features,survival)
    #train test split으로 나눈다
    print(train_features)
    scalar = StandardScaler()
    train_features = scalar.fit_transform(train_features)
    test_features = scalar.transform(test_features)
    print(train_features)
    model = LogisticRegression()
    model.fit(train_features, train_labels)

    print(model.score(train_features, train_labels))

    print(model.score(test_features,test_labels))

    Ahn = np.array([0,28,0,0,1])
    Kwon = np.array([1, 30, 0, 0,0])
    Who = np.array([1, 30, 0, 1,0])

    sample_passengers = np.array([Ahn,Kwon,Who])
    sample_passengers = scalar.transform(sample_passengers)
    print(model.predict(sample_passengers))

    print(model.predict_proba(sample_passengers))

if __name__ == "__main__":
    print("main")
    #titanic()
    regressionPrice()