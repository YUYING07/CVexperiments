import matplotlib.pyplot as plt
import numpy as np

#1-2 图像合成
image_path2 = './images/pro_1/a.png'
image_path3 = './images/pro_1/bg.png'
# 读取图片
img = plt.imread(image_path2)
img_bg = plt.imread(image_path3)
#alpha通道是所有图片 所有像素 的第四维
# print(img)
alpha = img[:, :, 3]
img_alpha = []
#得到alpha
#行代表一张图像的像素个数 共列数个
for i in range(alpha.shape[0]):  #逐个图像
    img_img = []
    for j in range(alpha.shape[1]): #图像每个像素 *3 这样才可以显示
        temp = []
        temp.append(alpha[i][j])
        temp.append(alpha[i][j])
        temp.append(alpha[i][j])
        img_img.append(temp)
    img_alpha.append(img_img)
img_alpha = np.array(img_alpha)
plt.imshow(img_alpha)
plt.show()
#得到前景
img_r = img[:, :, 0]
img_g = img[:, :, 1]
img_b = img[:, :, 2]
img_fg = []
for i in range(alpha.shape[0]):  #逐个图像
    img_img = []
    for j in range(alpha.shape[1]): #图像每个像素 *3 这样才可以显示
        temp = []
        temp.append(img_r[i][j])
        temp.append(img_g[i][j])
        temp.append(img_b[i][j])
        img_img.append(temp)
    img_fg.append(img_img)
img_fg = np.array(img_fg)

# print(1-img_alpha)
# alpha*img_fg+(1-alpha)*img_bg
img_fg = np.multiply(img_fg, img_alpha)
img_bg = np.multiply(img_bg, 1-img_alpha)
res_image = img_fg + img_bg
plt.imshow(res_image)
plt.show()
