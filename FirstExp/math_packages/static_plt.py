import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse , Rectangle
import numpy as np
import pandas as pd
from scipy.stats import f # F分布相关的api
from scipy.stats import norm # 标准正太的api
from scipy.stats import chi2 # 卡方分布的api
from scipy.stats import t # t分布的api
from math_packages import distance as d
from math_packages import static_test as test

#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)

def norm_QQ(X , axes , standardized = True):
    '''
    传进来一个一维数组,和要绘制的axes(plt的subplot,以及是否要正则化的bool值)
    '''
    temp = X
    if(X.ndim != 1):
        print('数组并非一维,请选用多元正太QQ图')
        exit(-1)
    norm_plist = norm.ppf((np.array(range(1,len(temp)+1))-0.5)/len(temp) , 0 , 1) #* 构造标准正太的分位数list
    if(standardized):
        temp = (temp - np.mean(temp))/np.sqrt(np.var(temp)) #*对X进行标准化，因为如果X~N(a,b),则X-a/sqrt(b) ~ N(0 , 1)
        inf = np.min([temp,norm_plist]) ; sup = np.max([temp,norm_plist])
        x = np.arange(inf , sup + 0.1 , 0.1)
        axes.plot(x , x , color = 'green') #* 绘制y=x标准线
    axes.scatter(norm_plist , np.sort(temp) , s = 9 , alpha = 0.6)