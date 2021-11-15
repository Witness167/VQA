from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(24, 10)) #定义figure，（1）中的1是什么
ax_all = HostAxes(fig, [0, 0, 0.9, 0.8]) #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1

#parasite addtional axes, share x
ax_yn = ParasiteAxes(ax_all, sharex=ax_all)
ax_number = ParasiteAxes(ax_all, sharex=ax_all)
ax_other = ParasiteAxes(ax_all, sharex=ax_all)

#append axes
ax_all.parasites.append(ax_yn)
ax_all.parasites.append(ax_number)
ax_all.parasites.append(ax_other)

#invisible right axis of ax_all
ax_all.axis['right'].set_visible(False)
ax_all.axis['top'].set_visible(False)
ax_yn.axis['right'].set_visible(True)
ax_yn.axis['right'].major_ticklabels.set_visible(True)
ax_yn.axis['right'].label.set_visible(True)

#set label for axis
ax_all.set_ylabel('All Accuracy (%)')
ax_all.set_xlabel('D', fontsize=24)
ax_yn.set_ylabel('Yes/No Accuracy (%)')
ax_number.set_ylabel('Number Accuracy (%)')
ax_other.set_ylabel('Other Accuracy (%)')

load_axisline = ax_number.get_grid_helper().new_fixed_axis
wear_axisline = ax_other.get_grid_helper().new_fixed_axis

ax_number.axis['right2'] = load_axisline(loc='right', axes=ax_number, offset=(100,0))
ax_other.axis['right3'] = wear_axisline(loc='right', axes=ax_other, offset=(200,0))

fig.add_axes(ax_all)


#画线
x = np.array([1, 2, 3, 4, 5, 6])
all = np.array([64.94, 66.11, 66.74, 67.21, 67.07, 66.98])
yn = np.array([82.45, 83.55, 84.24, 84.95, 84.83, 84.62])
number = np.array([44.51, 47.7, 48.28, 49.09, 48.87, 48.73])
other = np.array([57.04, 57.72, 58.31, 58.52, 58.39, 58.39])
weights = ['0', '1', '2', '4', '6', '8', '10']

curve_all, = ax_all.plot(x, all, label="All", color='black', marker='o', markersize=15, linewidth=5)
curve_yn, = ax_yn.plot(x, yn, label="Yes/No", color='red', marker='o', markersize=15, linewidth=5)
curve_number, = ax_number.plot(x, number, label="Number", color='green', marker='o', markersize=15, linewidth=5)
curve_other, = ax_other.plot(x, other, label="Other", color='blue', marker='o', markersize=15, linewidth=5)

ax_all.set_xticklabels(weights)
ax_all.set_ylim(64.50, 68)
ax_yn.set_ylim(82.00, 85.00)
ax_number.set_ylim(44.50, 49.50)
ax_other.set_ylim(57.00, 59.00)

ax_all.legend(fontsize=24, loc='lower right')

#轴名称，刻度值的颜色
#ax_all.axis['left'].label.set_color(ax_all.get_color())
ax_yn.axis['right'].label.set_color('red')
ax_number.axis['right2'].label.set_color('green')
ax_other.axis['right3'].label.set_color('blue')

ax_yn.axis['right'].major_ticks.set_color('red')
ax_number.axis['right2'].major_ticks.set_color('green')
ax_other.axis['right3'].major_ticks.set_color('blue')

ax_yn.axis['right'].major_ticklabels.set_color('red')
ax_number.axis['right2'].major_ticklabels.set_color('green')
ax_other.axis['right3'].major_ticklabels.set_color('blue')

ax_yn.axis['right'].line.set_color('red')
ax_number.axis['right2'].line.set_color('green')
ax_other.axis['right3'].line.set_color('blue')

ax_all.axis['left'].major_ticklabels.set_fontsize(24)
ax_yn.axis['right'].major_ticklabels.set_fontsize(24)
ax_number.axis['right2'].major_ticklabels.set_fontsize(24)
ax_other.axis['right3'].major_ticklabels.set_fontsize(24)

ax_all.axis['left'].label.set_fontsize(24)
ax_yn.axis['right'].label.set_fontsize(24)
ax_number.axis['right2'].label.set_fontsize(24)
ax_other.axis['right3'].label.set_fontsize(24)

ax_all.axis['bottom'].label.set_fontsize(24)
ax_all.axis['bottom'].major_ticklabels.set_fontsize(36)

plt.savefig('./yyy.jpg', bbox_inches='tight')
plt.show()