import numpy as np
import matplotlib.pyplot as plt

#图像缩放

def bilinear(img, length, height):
    length = 1/length
    height = 1/height
    x, y = int(img.shape[0]*length), int(img.shape[1]*height)
    res_img = np.zeros((x, y, 4), dtype=np.float)
    for i in range(x):
        for j in range(y):
            temp_x = int(i / length)
            temp_y = int(j / height)
            u = i / length - temp_x
            v = j / height - temp_y
            # 防止边缘越界
            m1 = min(temp_x + 1, img.shape[0] - 1)
            n1 = min(temp_y + 1, img.shape[1] - 1)
            res_img[i, j, :] = img[temp_x, temp_y, :] * (1 - u) * (1 - v) + img[m1, n1, :] * (1-u) * v + img[m1, temp_y, :] * u * (1-v) + img[temp_x, n1, :] * u * v
    return res_img


img_path = './images/pro_3/lab2.png'
source_img = plt.imread(img_path)
img = bilinear(source_img, 0.5, 1.5)
plt.subplot(1, 2, 1)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(source_img)
plt.subplot(1, 2, 2)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(img)
plt.show()
