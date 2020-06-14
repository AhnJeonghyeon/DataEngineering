import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    titanic()