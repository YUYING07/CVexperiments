import numpy as np
import matplotlib.pyplot as plt
import time

#图像变形
def distort_img(img_path):
    img = plt.imread(img_path)
    width, height = img.shape[0], img.shape[1]
    output_img = np.zeros((width, height, 4), dtype=np.float)
    # print(width, height)
    for i in range(width):
        for j in range(height):
            # 中心归一化
            temp_x = (i-0.5*width)/(0.5*width)
            temp_y = (j - 0.5*height)/(0.5*height)
            r = np.sqrt(temp_x**2 + temp_y**2)
            theta = (1-r)**2
            # 根据映射式子 得到映射值
            if r >= 1:
                x = temp_x
                y = temp_y
            else:
                x = np.cos(theta)*temp_x - np.sin(theta)*temp_y
                y = np.sin(theta)*temp_x + np.cos(theta)*temp_y
            #  从归一化的情况恢复
            x = int((x+1)*0.5*width)
            y = int((y+1)*0.5*height)
            output_img[i, j, :] = img[x, y, :]
    return output_img

img_path = './images/pro_3/lab2.png'
plt.subplot(1, 2, 1)
source_img = plt.imread(img_path)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(source_img)
plt.subplot(1, 2, 2)
res_img = distort_img(img_path)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(res_img)
plt.show()
