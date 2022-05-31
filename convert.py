import os
import numpy as np
import imageio

data = np.zeros((640000, 3), np.uint8)

f = open('2.ppm', encoding='utf-16')
i = 0
for line in f:
    pixel = line[:-1].split(' ')
    if len(pixel) < 3:
        continue

    data[i, 0] = int(pixel[0])
    data[i, 1] = int(pixel[1])
    data[i, 2] = int(pixel[2])

    i = i + 1

data.reshape((800, 800, 3))
imageio.imwrite('TheNextWeek.jpg', data)
