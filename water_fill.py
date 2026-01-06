# -*- coding: utf-8 -*-
"""
2021.01.29 注水法进行功率分配
注水法代码参考 https://pyphysim.readthedocs.io/en/latest/_modules/pyphysim/comm/waterfilling.html
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

def waterfilling(Channels, TotalPower, NoisePower):
    """ 注水算法进行功率分配
        Channels: 信道增益
        TotalPower: 待分配的总发射功率
        NoisePower: 接收端的噪声功率
    
    Returns:
        Powers: optimum powers (分配的功率)
        mu: water level (水位)
    """
    ### 降序排列信道增益
    Channels_SortIndexes = np.argsort(Channels)[::-1]
    Channels_Sorted = Channels[Channels_SortIndexes]
    """
    计算接触最差信道的水位，对这个最差的信道分配零功率。
    此后，按此水位为每个信道分配功率，
        如果功率之和少于总功率，则均分剩余功率给各个信道（增加水位）；
        如果功率之和多于总功率，则移除最坏信道，重复操作
    """
    N_Channels = Channels.size ## 总信道数
    N_RemovedChannels = 0  ## 移除的信道数
    ## 按最差信道计算最低水位
    WaterLevel = NoisePower / (Channels_Sorted[N_Channels-N_RemovedChannels-1])
    Powers = WaterLevel - (NoisePower /Channels_Sorted[np.arange(0, N_Channels - N_RemovedChannels)])
    
    ## 当功率之和多于总功率时，移除最坏信道，直至总功率够分
    while (sum(Powers)>TotalPower) and (N_RemovedChannels<N_Channels):
        N_RemovedChannels += 1
        WaterLevel = NoisePower / (Channels_Sorted[N_Channels-N_RemovedChannels-1])
        Powers = WaterLevel - (NoisePower /Channels_Sorted[np.arange(0, N_Channels - N_RemovedChannels)])
    
    ## 将剩余的功率均分给各个(剩余的)信道
    Power_Remine = TotalPower-np.sum(Powers)
    Powers_Opt_Temp = Powers + Power_Remine/(N_Channels - N_RemovedChannels)
    
    ## 将功率分配情况按原信道顺序排列
    Powers_Opt = np.zeros([N_Channels])
    Powers_Opt[Channels_SortIndexes[np.arange(0, N_Channels-N_RemovedChannels)]] = Powers_Opt_Temp
    
    WaterLevel = Powers_Opt_Temp[0] + NoisePower / Channels_Sorted[0]



    matplotlib.rcParams.update({'font.size': 14})
    buckets = Channels.size
    alpha = NoisePower/np.array(Channels)
    axis = np.arange(0.5,buckets+1.5,1)
    index = axis+0.5
    X = Powers_Opt.copy()
    Y = alpha+ X
    
    # to include the last data point as a step, we need to repeat it
    A = np.concatenate((alpha,[alpha[-1]]))
    X = np.concatenate((X,[X[-1]]))
    Y = np.concatenate((Y,[Y[-1]]))
    
    plt.xticks(index)
    plt.xlim(0.5,buckets+0.5)
    #    plt.ylim(0,1.5)
    plt.step(axis,A,where='post',label =r'$\sigma^2/h_i$',lw=2)
    plt.step(axis,Y,where='post',label=r'$\sigma^2/h_i + p_i$',lw=2)
    plt.legend(loc='upper left')
    plt.xlabel('Bucket Number')
    plt.ylabel('Power Level')
    plt.title('Water Filling Solution')
    plt.show()
    
    return Powers_Opt, WaterLevel

