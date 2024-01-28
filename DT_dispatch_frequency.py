import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import rcParams

# config = {
#     "font.family":'serif',
#     # "font.size": 20,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
# }
# rcParams.update(config)

def smooth_data(data, weight=0.95):
    smoothed_data = []
    last = data[0]
    for point in data:
        smoothed_point = last * weight + (1 - weight) * point
        smoothed_data.append(smoothed_point)
        last = smoothed_point
    return smoothed_data

go_num20 = 'logs/go_num/go_num20.npy'
go_num25 = 'logs/go_num/go_num25.npy'
go_num30 = 'logs/go_num/go_num30.npy'
go_num35 = 'logs/go_num/go_num35.npy'
go_num40 = 'logs/go_num/go_num40.npy'
go_num45 = 'logs/go_num/go_num45.npy'
go_num50 = 'logs/go_num/go_num50.npy'

# 计算每1000个数中出现次数
bins = range(0, 10001, 1000)


# 提取x和y坐标
# go_num20 = np.load(go_num20)
# go_num20_x = go_num20[:,0]
# print(go_num20_x)
# go_num20_y = go_num20[:, 1]

go_num25 = np.load(go_num25)
go_num25_x = go_num25[:,0]
go_num25_y = go_num25[:, 1]

go_num30 = np.load(go_num30)
go_num30_x = go_num30[:,0]
go_num30_y = go_num30[:, 1]

go_num35 = np.load(go_num35)
go_num35_x = go_num35[:,0]
go_num35_y = go_num35[:, 1]

go_num40 = np.load(go_num40)
go_num40_x = go_num40[:,0]
go_num40_y = go_num40[:, 1]

go_num45 = np.load(go_num45)
go_num45_x = go_num45[:,0]
go_num45_y = go_num45[:, 1]

go_num50 = np.load(go_num50)
go_num50_x = go_num50[:,0]
go_num50_y = go_num50[:, 1]

# go_num20_x_total, _ = np.histogram(go_num20_x, bins)
go_num25_x_total, _ = np.histogram(go_num25_x, bins)
go_num30_x_total, _ = np.histogram(go_num30_x, bins)
go_num35_x_total, _ = np.histogram(go_num35_x, bins)
go_num40_x_total, _ = np.histogram(go_num40_x, bins)
go_num45_x_total, _ = np.histogram(go_num45_x, bins)
go_num50_x_total, _ = np.histogram(go_num50_x, bins)

# smoothed_vals1 = smooth_data(go_num20_x_total)
smoothed_vals2 = smooth_data(go_num25_x_total)
smoothed_vals3 = smooth_data(go_num30_x_total)
# smoothed_vals4 = smooth_data(go_num35_x_total)
smoothed_vals5 = smooth_data(go_num40_x_total)
smoothed_vals6 = smooth_data(go_num45_x_total)
smoothed_vals7 = smooth_data(go_num50_x_total)

# # plt.plot(bins[:-1], smoothed_vals1)
# plt.plot(bins[:-1], smoothed_vals2)
# plt.plot(bins[:-1], smoothed_vals3)
# # plt.plot(bins[:-1], smoothed_vals4)
# plt.plot(bins[:-1], smoothed_vals5)
# plt.plot(bins[:-1], smoothed_vals6)
# plt.plot(bins[:-1], smoothed_vals7)

# 每50步进行一次标记
# marker_interval = 50

# plt.plot(bins[:-1][::marker_interval], smoothed_vals1[::marker_interval], linestyle='-', marker='o')
plt.plot(bins[:-1], smoothed_vals2, linestyle='--', marker='x')
plt.plot(bins[:-1], smoothed_vals3, linestyle='-.', marker='s')
# plt.plot(bins[:-1][::marker_interval], smoothed_vals4[::marker_interval], linestyle=':', marker='d')
plt.plot(bins[:-1], smoothed_vals5, linestyle='-', marker='+')
plt.plot(bins[:-1], smoothed_vals6, linestyle='--', marker='*')
plt.plot(bins[:-1], smoothed_vals7, linestyle='-.', marker='^')

plt.legend(['25', '30', '40', '45', '50'])

# all_count_x = ['1000', '25', '30', '40', '50']

# all_count_y = [sum(go_num20_y_counts), sum(go_num25_y_counts), sum(go_num30_y_counts), sum(go_num35_y_counts), sum(go_num40_y_counts), sum(go_num45_y_counts), sum(go_num50_y_counts)]
# all_count_y = [sum(go_num20_y_counts), sum(go_num25_y_counts), sum(go_num30_y_counts), sum(go_num40_y_counts), sum(go_num50_y_counts)]

# 绘制柱状图
# plt.bar(all_count_x, all_count_y)
plt.xlabel('Step Number')
plt.ylabel('UAV Dispatch Frequency')
# plt.xlabel('执行步数')
# plt.ylabel('无人机的派遣频率')
# plt.title('Comparison of y_count for Different Groups')
plt.show()
