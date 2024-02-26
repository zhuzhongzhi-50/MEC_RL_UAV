
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from matplotlib import rcParams

config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

def format_num(x, pos):
    return int(x * 2000)

# 得到一个表示圆形的点的集合，用于可视化设备的覆盖范围
def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r**2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]

# 指定要加载的 .npy 文件路径
uav_positoin_x0 = 'logs/position/uav_x0.npy'
uav_positoin_y0 = 'logs/position/uav_y0.npy'
uav_positoin_x1 = 'logs/position/uav_x1.npy'
uav_positoin_y1 = 'logs/position/uav_y1.npy'
uav_positoin_x2 = 'logs/position/uav_x2.npy'
uav_positoin_y2 = 'logs/position/uav_y2.npy'
uav_positoin_x3 = 'logs/position/uav_x3.npy'
uav_positoin_y3 = 'logs/position/uav_y3.npy'

# 加载 .npy 文件
x0 = np.load(uav_positoin_x0)
y0 = np.load(uav_positoin_y0)
x1 = np.load(uav_positoin_x1)
y1 = np.load(uav_positoin_y1)
x2 = np.load(uav_positoin_x2)
y2 = np.load(uav_positoin_y2)
x3 = np.load(uav_positoin_x3)
y3 = np.load(uav_positoin_y3)

# 计算每个点的颜色值
colors = np.linspace(0, 1, len(x0)) # 使用线性渐变色

# 创建新的图形窗口
fig, ax = plt.subplots(figsize=(9, 9))

# 在子图中绘制轨迹图
ax.plot(x0, y0, alpha=0.2)
sc = ax.scatter(x0, y0, s=5, c=colors, cmap='viridis', alpha=0.5)
ax.plot(x1, y1, alpha=0.2)
ax.scatter(x1, y1, s=5, c=colors, cmap='viridis', alpha=0.5)
ax.plot(x2, y2, alpha=0.2)
ax.scatter(x2, y2, s=5, c=colors, cmap='viridis', alpha=0.5)
ax.plot(x3, y3, alpha=0.2)
ax.scatter(x3, y3, s=5, c=colors, cmap='viridis', alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = plt.colorbar(sc, shrink=0.8)
formatter = FuncFormatter(format_num)
cbar.ax.yaxis.set_major_formatter(formatter)
cbar.set_label('步数')

# 添加文本标签
bbox_props = dict(boxstyle='round,pad=0.25', fc='white', ec='blue', lw=1, alpha=0.5)
# ax.text(0.22, 0.97, 'uav-2', transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
# ax.text(0.27, 0.15, 'uav-4', transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
# ax.text(0.02, 0.45, 'uav-1', transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
# ax.text(0.8, 0.35, 'uav-3', transform=ax.transAxes, fontsize=12,
#         verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
ax.text(0.22, 0.97, '无人机-2', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
ax.text(0.27, 0.15, '无人机-4', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
ax.text(0.02, 0.45, '无人机-1', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)
ax.text(0.8, 0.35, '无人机-3', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=bbox_props, color='blue', alpha=0.5)

# 在子图中绘制设备位置和场景车道
servers = ax.scatter(50, 50, s=8, c='orangered', alpha=0.9, label='Servers')
servers = ax.scatter(50, 150, s=8, c='orangered', alpha=0.9, label='Servers')
servers = ax.scatter(150, 50, s=8, c='orangered', alpha=0.9, label='Servers')
servers = ax.scatter(150, 150, s=8, c='orangered', alpha=0.9, label='Servers')

ax.annotate(1, xy=(50, 50), xytext=(50 + 0.1, 50 + 0.1))
ax.annotate(2, xy=(50, 150), xytext=(50 + 0.1, 150 + 0.1))
ax.annotate(3, xy=(150, 50), xytext=(150 + 0.1, 50 + 0.1))
ax.annotate(4, xy=(150, 150), xytext=(150 + 0.1, 150 + 0.1))

collect_plot1 = get_circle_plot((50,50), 30)
collect_plot2 = get_circle_plot((50,150), 30)
collect_plot3 = get_circle_plot((150,50), 30)
collect_plot4 = get_circle_plot((150,150), 30)

ax.fill_between(collect_plot1[0], collect_plot1[1], collect_plot1[2], where=collect_plot1[1] > collect_plot1[2], color='darkorange', alpha=0.05)
ax.fill_between(collect_plot2[0], collect_plot2[1], collect_plot2[2], where=collect_plot2[1] > collect_plot2[2], color='darkorange', alpha=0.05)
ax.fill_between(collect_plot3[0], collect_plot3[1], collect_plot3[2], where=collect_plot3[1] > collect_plot3[2], color='darkorange', alpha=0.05)
ax.fill_between(collect_plot4[0], collect_plot4[1], collect_plot4[2], where=collect_plot4[1] > collect_plot4[2], color='darkorange', alpha=0.05)

ax.fill_between([0,200], 75, 125, color='navy', alpha=0.1)
ax.fill_between([75,125], 0, 200, color='navy', alpha=0.1)

ax.grid()
# ax.legend(handles=[servers], labels=['Servers'], loc='upper right')
ax.legend(handles=[servers], labels=['服务器'], loc='upper right')
ax.axis('square')
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])

# plt.title('all uavs position epoch(0--2000)')
plt.title('所有无人机位置(0--2000)')

plt.show()
