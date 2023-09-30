import math

#Before starting the one of the most basic concept callled "Linear Regression(LR)" we can start with wonking of Linear_Regression just for understanding

#By using the program we can predict the marks

import pandas as pd
import numpy as np

def pred(x1,y1):
    m_curr, b_curr = 0,0
    ini = 0.0002
    prev = 0
    cost2 = 1
    n = len(x1)
    i = 0
    while(not(math.isclose(prev,cost2,rel_tol = 1e-10))):
        i+=1
        prev = cost2
        y_pred = m_curr*x1 + b_curr
        cost2 = (1/n)*sum(val**2 for val in (y1 - y_pred))
        md = -(2/n)*sum(x1*(y1 - y_pred))
        bd = -(2/n)*sum(y1 - y_pred)
        m_curr = m_curr - ini*md
        b_curr = b_curr - ini*bd
        print(f"The weight of {m_curr} and bias of {b_curr} with the cost of {cost2} in the iteration of {i}")
    return [m_curr,b_curr]
def calc(f,a):

    x = f['math']
    y = f['cs']
    x = np.array(x)
    y = np.array(y)

    m, b = pred(x, y)
    predicted_ans = math.floor(a*m + b)
    return predicted_ans

file = pd.read_csv('marks.csv')
predicted_value = calc(file,92)
print(f"When the math mark is 80 and :\nAuctal cs mark :83\npredicted cs mark {predicted_value}")









