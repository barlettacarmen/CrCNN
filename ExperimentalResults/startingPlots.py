import matplotlib.pyplot as plt
import math

x=[2,4,6,8,10]
y=[40411522,39756116,41012162,39833409,42022963]
y=[math.log2(i) for i in y]
xx=[2,4,8,16]
yy=[26,26,26,26]
y2=[21772057,47601119,41749972,93155811]
y2=[math.log2(i) for i in y2]
plt.xlabel('#imgs_for_each_binary_search_run')
plt.ylabel('log2_of_plain_mod_found')
plt.plot(x,y,label='no-pow')
plt.plot(xx,y2,label='no-pow')
plt.plot(xx,yy,label='pow')
plt.legend()
plt.show()