import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sys
from math import log2
import seaborn as snb

name_ar=["BATCH_SIZE","PLAIN_MOD","TIME"]
df=pd.read_csv(sys.argv[1],header=None,sep=',',names=name_ar)

#print(df)
plt.show()
for k,subdf in df.groupby(["BATCH_SIZE"]):
	#plt.scatter( [k for i in range(len(subdf)) ], [log2(el) for el in subdf['PLAIN_MOD']], label=str(k) )
	plt.scatter( [k for i in range(1)] , [log2(np.max(subdf['PLAIN_MOD']))],label=str(k) )
	#plt.scatter( [k for i in range(2)] , [log2(np.max(subdf['PLAIN_MOD'])), log2(np.mean(subdf['PLAIN_MOD']))],label=str(k) )
	#plt.scatter( [k for i in range(3)] , [log2(np.min(subdf['PLAIN_MOD'])),log2(np.mean(subdf['PLAIN_MOD'])),log2(np.max(subdf['PLAIN_MOD']))] , label=str(k) )
	print(k,'. n elems:', len(subdf))

# for k,subdf in df.groupby(["BATCH_SIZE"]):
# 	#plt.scatter( [k for i in range(len(subdf)) ], [log2(el) for el in subdf['PLAIN_MOD']], label=str(k) )
# 	plt.scatter( np.mean(subdf['TIME']), log2(np.max(subdf['PLAIN_MOD'])), label=str(k) )
# 	print(k,'. n elems:', len(subdf))

plt.xlabel("BATCH_SIZE")
plt.ylabel("log2(max(PLAIN_MOD_FOUND))")
plt.xscale('log',basex=2)
plt.legend()
plt.show()
