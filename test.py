# import glob
# import imageio
# # 生成 gif 格式的图片
# env_log_dir = 'logs/picture/picture20231018-171358'
# img_paths = glob.glob(env_log_dir + '/*.png')
# img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))
# gif_images = []
# for path in img_paths:
#     gif_images.append(imageio.imread(path))
# imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=20)


import numpy as np
dispatch_epoch = 'logs/dispatch_epoch.npy'

# 提取x和y坐标
dispatch_epoch_num = np.load(dispatch_epoch)
print(dispatch_epoch_num)