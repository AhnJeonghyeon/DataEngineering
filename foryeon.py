'''
영향을 주는 요소
1. 최종합격 TO 수
2. 지원자 수
3. 점수
input 합격점수 output합격여부

'''

import pandas as pd
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 합격자수가 num이고 지원이 total일 때 realnum만큼 응시하고 합격점수는 score이다
    df = pd.read_csv('/Users/AhnJeongHyeon/Desktop/혼자공부/tensorflow/ForYeon/data.csv')
    print(df)
    train_data = df[0:22]
    train_test = df[23:24]

    model1 = linear_model.LinearRegression()
    x_vars1 = ['num', 'total']
    model1.fit(train_data[x_vars1], train_data['score'])
    print(model1.coef_, model1.intercept_)

    x=np.linspace(1,10,23)

    df['num by total'] = df['num'] * df['total']
    model2 = linear_model.LinearRegression()
    x_var2 = ['num', 'total', 'num by total']
    model2.fit(train_data[x_var2], train_data['realnum'])
    print(model2.coef_, model1.intercept_)

    plt.scatter(train_data['num'], train_data['total'], c=train_data['realnum'])
    plt.plot(x, x * model2.coef_[0] + model2.coef_[2] + model2.intercept_)
    plt.plot(x, x * model2.coef_[0] + model2.intercept_)
    plt.show()

    print(np.mean((model2.predict(train_test[x_vars1]) - train_test['score']) ** 2))
'''
    plt.scatter(train_data['num'],train_data['total'], c=train_data['score'])
    plt.plot(x, x*model1.coef_[0]+model1.coef_[1]+model1.intercept_)
    plt.plot(x, x*model1.coef_[0]+model1.intercept_)
    plt.show()
'''
