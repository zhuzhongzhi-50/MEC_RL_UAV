import numpy as np

# 车道内的车辆传感器的移动规则设计
def traffic_move(sensor, move, i): 
    # 左下角区域
    if sensor.position[0] <= 75 and sensor.position[1] <= 75:
        sensor.position[0] += move[0] 
        sensor.position[1] += move[1]
        # 防止移动出界，如果移动出界，则不进行移动
        if  sensor.position[0] >= 75 or sensor.position[0] <= 0:
            sensor.position[0] -= move[0] 
        if  sensor.position[1] >= 75 or sensor.position[1] <= 0:
            sensor.position[1] -= move[1]
        # 只做一种情况的 判断，满足就直接跳出
        return True
    # 左上角区域
    if sensor.position[0] <= 75 and sensor.position[1] >= 125:
        sensor.position[0] += move[0] 
        sensor.position[1] += move[1]
        if  sensor.position[0] >= 75 or sensor.position[0] <= 0:
            sensor.position[0] -= move[0] 
        if  sensor.position[1] <= 125 or sensor.position[1] >= 200:
            sensor.position[1] -= move[1]
        return True
    # 右下角区域 
    if sensor.position[0] >= 125 and sensor.position[1] <= 75:
        sensor.position[0] += move[0] 
        sensor.position[1] += move[1]
        if  sensor.position[0] < 125 or sensor.position[0] >= 200:
            sensor.position[0] -= move[0] 
        if  sensor.position[1] > 75 or sensor.position[1] <= 0:
            sensor.position[1] -= move[1]
        return True
    # 右上角区域   
    if sensor.position[0] >= 125 and sensor.position[1] >= 125:
        sensor.position[0] += move[0] 
        sensor.position[1] += move[1]
        if  sensor.position[0] < 125 or sensor.position[0] >= 200:
            sensor.position[0] -= move[0] 
        if  sensor.position[1] < 125 or sensor.position[1] >= 200:
            sensor.position[1] -= move[1]
        return True
    # 左下横车道，0、100 都需要覆盖到
    if sensor.position[0] >= 0 and sensor.position[0] < 95 and sensor.position[1] > 75 and sensor.position[1] <= 100:
        if move[0] > 0:
            sensor.position[0] += move[0]
        if move[0] < 0:
            sensor.position[0] -= move[0]
        if sensor.position[0] >= 95:
            sensor.position[0] = 95
        return True
    # 左上横车道
    if sensor.position[0] >= 0 and sensor.position[0] < 95 and sensor.position[1] > 100 and sensor.position[1] < 125:
        if move[0] > 0:
            sensor.position[0] -= move[0]
        if move[0] < 0:
            sensor.position[0] += move[0]
        if sensor.position[0] <= 0:
            sensor.position[1] -= 25
            sensor.position[0] = 1
        return True
    # 右下横车道
    if sensor.position[0] > 105 and sensor.position[0] <= 200 and sensor.position[1] > 75 and sensor.position[1] <= 100:
        if move[0] > 0:
            sensor.position[0] += move[0]
        if move[0] < 0:
            sensor.position[0] -= move[0]
        if sensor.position[0] >= 200:
            sensor.position[1] += 25
            sensor.position[0] = 199
        return True
    # 右上横车道
    if sensor.position[0] > 105 and sensor.position[0] <= 200 and sensor.position[1] > 100 and sensor.position[1] < 125:
        if move[0] > 0:
            sensor.position[0] -= move[0]
        if move[0] < 0:
            sensor.position[0] += move[0]
        if sensor.position[0] <= 105:
            sensor.position[0] = 105
        return True
    # 左下纵车道
    if sensor.position[0] > 75 and sensor.position[0] < 100 and sensor.position[1] >= 0 and sensor.position[1] < 75:
        if move[1] > 0:
            sensor.position[1] -= move[1]
        if move[1] < 0:
            sensor.position[1] += move[1]
        if sensor.position[1] <= 0:
            sensor.position[0] += 25
            sensor.position[1] = 1
        return True
    # 右下纵车道
    if sensor.position[0] >= 100 and sensor.position[0] < 125 and sensor.position[1] >= 0 and sensor.position[1] < 75:
        if move[1] > 0:
            sensor.position[1] += move[1]
        if move[1] < 0:
            sensor.position[1] -= move[1]
        if sensor.position[1] >= 75:
            sensor.position[1] = 75
        return True
    # 左上纵车道
    if sensor.position[0] > 75 and sensor.position[0] < 100 and sensor.position[1] > 125 and sensor.position[1] <= 200:
        if move[1] > 0:
            sensor.position[1] -= move[1]
        if move[1] < 0:
            sensor.position[1] += move[1]
        if sensor.position[1] <= 125:
            sensor.position[1] = 125
        return True
    # 右上纵车道
    if sensor.position[0] >= 100 and sensor.position[0] < 125 and sensor.position[1] > 125 and sensor.position[1] <= 200:
        if move[1] > 0:
            sensor.position[1] += move[1]
        if move[1] < 0:
            sensor.position[1] -= move[1]
        if sensor.position[1] >= 200:
            sensor.position[0] -= 25
            sensor.position[1] = 199
        return True
    
    # 中心区域处理
    # 左下
    if sensor.position[0] >= 95 and sensor.position[0] <= 100 and sensor.position[1] >= 75 and sensor.position[1] <= 100:
        a = np.random.randint(1,5)
        if a == 1:
            sensor.position[1] += 25
            sensor.position[0] = 74
        if a == 2:
            sensor.position[0] = sensor.position[1] + 25
            sensor.position[1] = 126
        if a == 3:
            sensor.position[0] = 126
        if a == 4:
            sensor.position[0] = sensor.position[1]
            sensor.position[1] = 74
        return True

    # 左上
    if sensor.position[1] == 125  and sensor.position[0] >= 75 and sensor.position[0] <= 100:
        a = np.random.randint(1,5)
        if a == 1:
            sensor.position[0] += 25
            sensor.position[1] = 126
        if a == 2:
            sensor.position[1] = sensor.position[0]
            sensor.position[0] = 126
        if a == 3:
            sensor.position[1] = 74
        if a == 4:
            sensor.position[1] = sensor.position[0]
            sensor.position[0] = 74
        return True

    # 右上
    if sensor.position[0] >= 100 and sensor.position[0] <= 105 and sensor.position[1] >= 100 and sensor.position[1] <= 125:
        a = np.random.randint(1,5)
        if a == 1:
            sensor.position[1] -= 25
            sensor.position[0] = 126 
        if a == 2:
            sensor.position[0] = sensor.position[1] - 25
            sensor.position[1] = 74
        if a == 3:
            sensor.position[0] = 74
        if a == 4:
            sensor.position[0] = sensor.position[1]
            sensor.position[1] = 126
        return True

    # 右下
    if sensor.position[1] == 75 and sensor.position[0] >= 100 and sensor.position[0] <= 125:
        a = np.random.randint(1,5)
        if a == 1:
            sensor.position[0] -= 25
            sensor.position[1] = 74
        if a == 2:
            sensor.position[1] = sensor.position[0]
            sensor.position[0] = 74
        if a == 3:
            sensor.position[1] = 126
        if a == 4:
            sensor.position[1] = sensor.position[0]-25
            sensor.position[0] = 126
        return True