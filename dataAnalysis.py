import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def logisticRegressionPrice():
    #todo 집값 예측 모듈 만들어야돼
    df = pd.read_csv('Data/APT19.06~20.0530.csv')

    #df = pd.read_csv('Data/test.csv')
    df.columns = ['location', 'fnum', 'num', 'semiNum', 'name', 'size', 'ym', 'd', 'price', 'floor', 'buildY',
                  'address']
    #평 평당 가격 추가

    #***자료형 변형
    df['pyung'] = df['size'] / (3.3)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    #df['price'] = df['price'].astype(float)
    df['p_price'] = df['price']/df['pyung']
    df['result'] = 0

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
    features = last[['location','pyung','buildY','price']]
    #features = last[['location', 'pyung', 'price']]
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
    test = np.array([1,1.0,1995,70000])
    test2 = np.array([1,0.0,1995,50000])
    sample = np.array([test,test2])
    sample = scalar.fit_transform(sample)
    print(model.predict(sample))
    print(model.predict_proba(sample))


def algebraRegressionPrice():
    # todo 집값 예측 모듈 만들어야돼, linear, multi, polynomial정확성 비교해보자
    #df = pd.read_csv('Data/test.csv')
    df = pd.read_csv('Data/APT19.06~20.0530.csv')

    df.columns = ['location', 'fnum', 'num', 'semiNum', 'name', 'size', 'ym', 'd', 'price', 'floor', 'buildY',
                  'address']
    # 평 평당 가격 추가

    # ***자료형 변형
    df['pyung'] = df['size'] / (3.3)
    df['pyung'] = pd.to_numeric(df['pyung'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    #전체 y,m,d로 나눈다. (현재연도2020 - df['y'])*12 * -1 +
    # - (현재월 - df['m'])
    sys_y = 2020
    sys_m = 6
    df['y'] = (df['ym']/100)
    df['y'] = df['y'].astype(int)
    df['m'] = df['ym']%100
    df['ym'] = (sys_y-df['y'])*-12 - (sys_m-df['m'])
    # TODO regression
    '''
    algebra regression은 logistic과 다르게 데이터 자체를 잘라내는 작업이 필요하다
    강남이면 근본데이터에서 강남만 남기는 방식으로
    model에 들어가야 할 최적의 데이터만을 남겨놓고 그것을 이용해 predict하는 방식
    
       linear regression
       1. 지역을 입력받는다, 그 지역으로 필터링한다.
       2. 평수를 입력한다. 그 평수 +-5로 필터링한다 
       3. buildY를 입력한다 그 연도 +-5로 필터링한다
       4. 현재 집값을 입력한다. 집값이 0~5억인 경우 +-1억, 5~10 -> +-1.5, 10~ -> +-2
       
       x축은 시간(ym), y축은 집값(price)이다.
       
       결과) n년 뒤 집값을 예측한다.
    '''

    area, pyung, buildY, price = input("지역, 평수, 지어진 연도, 가격").split()
    pyung = int(pyung)
    buildY = int(buildY)
    price = int(price)
    temp = df[['location', 'pyung', 'buildY', 'price', 'ym']]

    list=[]
    for index, row in temp.iterrows():
        if row['location'].split()[2] == area:
            if pyung - 5 <= row['pyung']  <= pyung + 5 :
                if buildY-5 <= row['buildY'] <= buildY + 5:
                    if price < 50000:
                        if price-10000 <= row['price'] <= price+10000:
                            list.append(row)
                    elif price < 100000:
                        if price - 10000 <= row['price'] <= price + 10000:
                            list.append(row)
                    else:
                        if price - 20000 <= row['price'] <= price + 20000:
                            list.append(row)

    last = pd.DataFrame(list)

    #개포동 24 1984 150000
    #하계동 33 1995 70000
    #세로 2차원 리스트로 만들어야한다.
    x = np.array(last.T.loc['ym']).reshape((-1,1))
    y = np.array(last.T.loc['price'])

    print(last)
    '''
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    y = scalar.fit_transform(y)
    print(x)
    '''

    linearModel = LinearRegression()
    linearModel.fit(x,y)
    r_sq = linearModel.score(x,y)
    print(r_sq)
    print(linearModel)
    y_pred = linearModel.predict([[1],[2]])

    print(y_pred)
    visualization(x,y,linearModel)
######################################polynomial Regression
    transformer = PolynomialFeatures(degree = 2, include_bias = False)
    transformer.fit(x)
    x_ = transformer.transform(x)
    print(x_)

    polynomialModel = LinearRegression().fit(x_,y)
    r_sq =polynomialModel.score(x_,y)
    print(r_sq)

    y_pred = polynomialModel.predict([[1],[2]])
    print(y_pred)



def visualization(x, y, model):
    plt.scatter(x, y, color="red")
    #plt.plot(x, model.predict([[201912]]), color="green")
    plt.title("housing value (Training set)")
    plt.xlabel("ym")
    plt.ylabel("price")
    plt.show()


if __name__ == "__main__":
    print("main")
    #logisticRegressionPrice()
    algebraRegressionPrice()