import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys
from math import log2
import seaborn as snb

name_ar=["I","T1","TR1","T2","TR2","T3","TR3","T4","TR4","T5","TR5","T6","TR6","T7","TR7","T8","TR8","T9","TR9","T10","TR10","T_ENC","T_DEC","PREDICTION"]
df=pd.read_csv(sys.argv[1],index_col=False,sep=',',names=name_ar)
print(df)
print(df.describe())