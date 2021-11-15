
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

#plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#x = np.array([1, 2, 3, 4])
#h = np.array([67.07, 67.08, 67.13, 67.21])

# all
# base = np.array([64.82, 65.76, 66.18, 67])
# self = np.array([65.03, 65.83, 66.27, 67.08])
# cross = np.array([65.13, 65.83, 66.22, 67.13])
# fumin = np.array([65.09, 66.11, 66.74, 67.21])

# other
# base = np.array([56.5, 57.1, 57.41, 58.49])
# self = np.array([56.6, 57.3, 57.38, 58.52])
# cross = np.array([56.7, 57.3, 57.39, 58.45])
# fumin = np.array([56.8, 57.7, 58.31, 58.52])

# #yes
# base = np.array([82.3, 83.5, 83.81, 84.8])
# self = np.array([82.4, 83.3, 84.11, 84.8])
# cross = np.array([82.6, 83.3, 84.05, 84.8])
# fumin = np.array([82.5, 83.6, 84.24, 85])

#number
# base = np.array([46, 47.4, 48.63, 48.65])
# self = np.array([47, 47.8, 48.58, 48.48])
# cross = np.array([46.9, 48, 48.28, 49.05])
# fumin = np.array([46.6, 47.7, 48.28, 49.09])

#all = np.array([66.38, 67.02, 67.25, 67.19, 67.18, 67.07, 66.66, 66.55, 66.46, 66.2, 64])
other = np.array([58.19, 58.52, 58.71, 58.59, 58.64, 58.49, 58.16, 58.17, 57.97, 57.79, 49.96])
#yn = np.array([84.51,84.6,84.79,84.63,84.78,84.76,84.4,83.98,84.01,83.78,69.53])
#number = np.array([45.23,48.61,49.09,49.01,48.89,48.65,47.8,48.15,48.15,47.47,37.08])
#VGG_unsupervised = np.array([2.1044724, 2.9757383, 3.7754183, 5.686206, 8.367847, 14.144531])
#ourNetwork = np.array([2.0205495, 2.6509762, 3.1876223, 4.380781, 6.004548, 9.9298])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(12, 8))
#plt.grid(linestyle="--", axis='y')  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

plt.plot(x, other, marker='o', color="b", markersize=12, linewidth=5, label="unbalanced")
plt.hlines(y=58.52, xmin=0, xmax=11, colors='r', label='DUIM (ours)', linewidth=8, linestyles='--')

# plt.plot(x, base, marker='^', color="k", markersize=15, linewidth=5, label="base")
# plt.plot(x, self, marker='^', color="g", markersize=15, linewidth=5, label="base+self")
# plt.plot(x, cross, marker='^', color="b", markersize=15, linewidth=5, label="base+cross")
# plt.plot(x, fumin, marker='^', color="r", markersize=15, linewidth=5, label="FUMIN")
#plt.plot(x, VGG_supervised, marker='o', color="b", linewidth=5, markersize=15)
#plt.plot(x, ourNetwork, marker='o', color="red", label="ShuffleNet-style Network", linewidth=1.5)

weights = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
#weights = ['1', '2', '4', '6']  # x轴刻度的标识
plt.xticks(x, weights, fontsize=24, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=24, fontweight='bold')
#plt.title("(a) All", fontsize=24, fontweight='bold', y=-0.2)  # 默认字体大小为12
#plt.title("", fontsize=24, fontweight='bold', y=-0.2)  # 默认字体大小为12
plt.xlabel("weight", fontsize=24, fontweight='bold')
plt.ylabel("Accuracy(%)", fontsize=24, fontweight='bold')
plt.xlim(0.7, 11.3)  # 设置x轴的范围
#plt.ylim(66.2, 67.5)
#plt.ylim(47.25, 49.25)
#plt.ylim(83.50, 85.50)
plt.ylim(57.5, 59)


plt.legend()          #显示各曲线的图例
plt.legend(loc=3, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=24, fontweight='bold')  # 设置图例字体的大小和粗细

#plt.savefig('unbalanceAll.jpg', bbox_inches='tight')
plt.savefig('unbalanceOther.jpg', bbox_inches='tight')
plt.show()