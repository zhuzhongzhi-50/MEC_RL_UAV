
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

dispatch_epoch = 'logs/go_num_plus/go_num60_25_plusdispatch_epoch.npy'
# 提取x和y坐标
dispatch_epoch_num = np.load(dispatch_epoch)
print(dispatch_epoch_num)

go_num60_25 = 'logs/go_num_plus/go_num60_25.npy'
go_num60_25_plus = 'logs/go_num_plus/go_num60_25_plus.npy'
go_num80_25 = 'logs/go_num_plus/go_num80_25.npy'
go_num80_25_plus = 'logs/go_num_plus/go_num80_25_plus2.npy'
go_num100_25 = 'logs/go_num_plus/go_num100_25.npy'

# 提取x和y坐标
go_num60_25 = np.load(go_num60_25)
go_num60_25_x = go_num60_25[:,0]
go_num60_25_y = go_num60_25[:,1]

go_num60_25_plus = np.load(go_num60_25_plus)
go_num60_25_plus_x = go_num60_25_plus[:,0]
go_num60_25_plus_y = go_num60_25_plus[:,1]
print(go_num60_25_plus_y)

go_num80_25 = np.load(go_num80_25)
go_num80_25_x = go_num80_25[:,0]
go_num80_25_y = go_num80_25[:,1]

go_num80_25_plus = np.load(go_num80_25_plus)
go_num80_25_plus_x = go_num80_25_plus[:,0]
go_num80_25_plus_y = go_num80_25_plus[:,1]


go_num100_25 = np.load(go_num100_25)
go_num100_25_x = go_num100_25[:,0]
go_num100_25_y = go_num100_25[:,1]

go_num60_25_y_size, go_num60_25_y_counts = np.unique(go_num60_25_y, return_counts=True)
go_num60_25_plus_y_size, go_num60_25_plus_y_counts = np.unique(go_num60_25_plus_y, return_counts=True)
go_num80_25_y_size, go_num80_25_y_counts = np.unique(go_num80_25_y, return_counts=True)
go_num80_25_plus_y_size, go_num80_25_plus_y_counts = np.unique(go_num80_25_plus_y, return_counts=True)
go_num100_25_y_size, go_num100_25_y_counts = np.unique(go_num100_25_y, return_counts=True)

plt.bar(go_num60_25_y_size, go_num60_25_y_counts, color='b', alpha=0.9, label='Data 1')
plt.bar(go_num60_25_plus_y_size, go_num60_25_plus_y_counts, color='r', alpha=0.7, label='Data 1')
# plt.bar(go_num80_25_y_size, go_num80_25_y_counts, color='r', alpha=0.7, label='Data 2')
# plt.bar(go_num80_25_plus_y_size, go_num80_25_plus_y_counts, color='b', alpha=0.7, label='Data 2')
# plt.bar(go_num100_25_y_size, go_num100_25_y_counts, color='y', alpha=0.7, label='Data 3')
plt.xlabel('Number')
plt.ylabel('Count')
plt.title('Count of each number')
plt.legend()
plt.show()