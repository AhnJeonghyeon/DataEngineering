import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def regressionPrice():
    #todo 집값 예측 모듈 만들어야돼
    df = pd.read_csv('Data/APT19.06~20.0530.csv')

    #df = pd.read_csv('Data/test.csv')
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
    temp = df[['location','name','pyung','buildY','result','price','ym']]
    temp2 = df[['location', 'name', 'pyung', 'buildY', 'result', 'price','ym']]
    temp3 = df[['location', 'name', 'pyung', 'buildY', 'result', 'price', 'ym']]

    value = int(input("값을 입력하세요"))

    '''
    value이상 오를지 안오를지 판단하기
    1. location,name,pyung을 하나의 row로 만들어야한다
        - groupby location, name, pyung, buildY, result ['price'].max()
        하나더 만들어서 min()으로
        둘의 len을 비교한 후 max-min값으로 재설정해준다 반복문 돌려서
    2. 강남/강북여부를 결정해준다
        -
    3. 30평대 1/0을 결정해준다
    4. BuildY
    4. 모델링 
        - 지역, pyung, buildY, 현재가격을 입력하면 value이상 오를지 알려줌
    '''
    # temp = temp.groupby(['location','name','pyung','buildY','result'])['price'].max()
    temp = pd.DataFrame(
        {'price': temp.groupby(['location', 'name', 'pyung', 'buildY', 'result'])['price'].max()}).reset_index()
    temp2 = pd.DataFrame(
        {'price': temp2.groupby(['location', 'name', 'pyung', 'buildY', 'result'])['price'].min()}).reset_index()
    temp3 = pd.DataFrame(
        {'ym': temp3.groupby(['location', 'name', 'pyung', 'buildY', 'result', 'price'])['ym'].max()}).reset_index()
    temp3 = temp3.sort_values(by=['location', 'name', 'pyung', 'buildY', 'result', 'ym'])
    temp3 = temp3.drop_duplicates(['location', 'name', 'pyung', 'buildY', 'result'],keep="first").reset_index(drop=True)
    #goupby를 df로 만듦

    temp['diff_price'] = temp['price'] - temp2['price']
    temp['price'] = temp3['price']
    temp['price'] = pd.to_numeric(temp['price'], errors='coerce')
    temp['diff_price'] = pd.to_numeric(temp['diff_price'], errors='coerce')

    #이제 강남강북, 30평대를 확인하여 값을 세팅한다.
    for index, row in temp.iterrows():
        if row['location'].split()[1] in gangNam:
            temp.loc[index,"location"] = 0
        else:
            temp.loc[index, "location"] = 1
        if 30.0 <= row['pyung'] < 50:
            temp.loc[index, "pyung"] = 1
        else:
            temp.loc[index, "pyung"] = 0
        if row['diff_price'] >= value:
            temp.loc[index, "result"] = 1
        else:
            temp.loc[index, "result"] = 0

    # 데이터 가공 완료
    print(temp.head(10))
    print(temp.tail(10))

    last = temp[['location', 'pyung', 'price', 'buildY','result']]

    #modeling
    #features = last[['location','pyung','buildY','price']]
    features = last[['location', 'pyung', 'price']]
    result = last['result']
    train_features, test_features, train_labels, test_labels = train_test_split(features,result)

    scalar = StandardScaler()
    train_features = scalar.fit_transform(train_features)
    test_features = scalar.fit_transform(test_features)

    model = LogisticRegression()
    model.fit(train_features, train_labels)
    print("model score")
    print(model.score(train_features, train_labels))
    print(model.score(test_features,test_labels))

    #location, pyung, diff_price,BuildY

    #test정보를 입력
    print("test start")
    test = np.array([1,1.0,70000])
    test2 = np.array([1,0.0,50000])
    sample = np.array([test,test2])
    print(model.predict(sample))
    print(model.predict_proba(sample))


if __name__ == "__main__":
    print("main")
    regressionPrice()