import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


# 定义散点数据
x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160])
y = np.array([1, 2, 3.5, 4, 4.5, 5, 6, 7.5, 7.7, 8, 8, 8, 8, 8, 8, 8])

# 绘制原始散点图
# plt.scatter(x, y, label='Data')

# 进行平滑处理
spl = make_interp_spline(x, y, k=3)  # 使用三次样条插值
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spl(x_smooth)

# 绘制平滑曲线
plt.plot(x_smooth, y_smooth, label='Smoothed', color='r')

# 绘制收敛直线效果
# plt.plot(x[-5:], y[-5:], linestyle='--', label='Converging Line', color='g')

# 添加图例、标签等
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Smoothing and Converging Line')
plt.show()
