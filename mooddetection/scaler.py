from sklearn.preprocessing import MinMaxScaler

data = [[.2], [.3], [.8]]
scaler = MinMaxScaler()#feature_range=(0.2,.8))

print( scaler.fit_transform(data))

import numpy as np
def convert(x,a,b,c=0,d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/float(b-a)

print(convert(.9, .2, .8, 0, 1))




