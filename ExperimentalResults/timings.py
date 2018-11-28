import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys
from math import log2
import seaborn as snb

#name_ar=["I","T1","T2","T3","T4","T5","TR","T6","T7","T8","T9","PREDICTION"]
name_ar=["I","T1","T2","T3","T4","TR","T5","T6","PREDICTION"]
df=pd.read_csv(sys.argv[1],index_col=False,sep=',',names=name_ar,)

print(df.describe())

#df["TOT"]=df[["T1","T2","T3","T4","T5","TR","T6","T7","T8","T9"]].sum(axis=1)
df["TOT"]=df[["T1","T2","T3","T4","TR","T5","T6"]].sum(axis=1)
print(np.mean(df["TOT"])*10**-6)


snb.distplot(df["TOT"])
plt.show()

# time=0
# var=0
# for c in range(1,11):
# 	print(df.ix[:,c].name)
# 	time+=np.mean(df.ix[:,c])

# print(time*10**-6)