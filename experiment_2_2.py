import numpy as np
import matplotlib.pyplot as plt

def jbf(image_D, image_C, w, sigma_f, sigma_g):

    res_image = image_D.copy()
    distance = np.zeros([w, w], dtype=np.float)
    # 算出滤波窗口内的距离
    for m in range(w):
        for n in range(w):
            distance[m, n] = (m - w//2) ** 2 + (n - w//2) ** 2

    for i in range(w//2, image_C.shape[0] - w//2):
        for j in range(w//2, image_C.shape[1] - w//2):
            for d in range(3):
                # 计算当前窗口范围
                istart = i - w//2
                iend = i + w//2
                jstart = j - w//2
                jend = j + w//2
                # 原图的当前窗口
                window_s = image_D[istart:iend + 1, jstart: jend + 1, d]
                # 引导图的当前窗口
                window_g = image_C[istart:iend + 1, jstart: jend + 1, d]
                # 由引导图像的灰度值差计算值域核
                f = np.exp(-0.5 * distance / (sigma_f ** 2))
                g = np.exp(-0.5 * (window_g - image_C[i, j, d]) ** 2 / (sigma_g ** 2))
                # 根据公式给出
                res_image[i, j, d] = np.sum(g * f * window_s) / np.sum(g * f)
    # print(res_image)
    return res_image


def bilinear(img, rate):
    # 双线性插值
    x, y = int(img.shape[0]*rate), int(img.shape[1]*rate)
    res_img = np.zeros((x, y, 4), dtype=np.float)
    for i in range(x):
        for j in range(y):
            temp_x = int(i / rate)
            temp_y = int(j / rate)
            u = i / rate - temp_x
            v = j / rate - temp_y
            # 防止边缘越界
            m1 = min(temp_x + 1, img.shape[0] - 1)
            n1 = min(temp_y + 1, img.shape[1] - 1)
            # 双线性插值的式子
            res_img[i, j, :] = img[temp_x, temp_y, :] * (1 - u) * (1 - v) + img[m1, n1, :] * (1-u) * v + img[m1, temp_y, :] * u * (1-v) + img[temp_x, n1, :] * u * v
    return res_img


#得到引导图像
image_path1 = './images/pro_2/b.png'
plt.subplot(1, 3, 1)
source_img = plt.imread(image_path1)
plt.imshow(source_img)
plt.subplot(1, 3, 2)
img = bilinear(source_img, 1/2)
img_2 = bilinear(img, 2)
plt.imshow(img_2)
# 进行联合双边滤波
plt.subplot(1, 3, 3)
res_image = jbf(source_img, img_2, 9, 11, 3)
plt.imshow(res_image)
plt.show()
