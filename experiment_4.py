import numpy as np
import cv2
from skimage import color, measure
import matplotlib.pyplot as plt

image_path = './images/pro_4/Orical1.jpg'
image_path2 = './images/pro_4/Orical2.jpg'


def BGR_to_RGB(image):
    # 由BGR转换成RGB
    img1 = np.array(image)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            temp = img1[i][j][0]
            img1[i][j][0] = img1[i][j][2]
            img1[i][j][2] = temp
    return img1

def findeyes(img_skin, minr, minc, maxr, maxc, img):
    '''

    :param img_skin: 进行开操作后的图像
    :param minr: 脸部区域的最小行
    :param minc: 脸部最小列
    :param maxr: 脸部最大行
    :param maxc: 脸部最大列
    :param img: 原图像
    :return: 是否有眼睛和 框定眼睛的图像
    '''

    # 如果区域内有两个以上的空框是眼睛
    part = np.zeros(((maxr - minr), (maxc - minc)))

    # 二值取反 原本图像中眼睛是黑色的
    for i in range(minr, maxr):
        for j in range(minc, maxc):
            if img_skin[i, j] == 0:
                part[i - minr, j - minc] = 255
            else:
                part[i - minr, j - minc] = 0

    # 标定连通区域
    part_labeled, num = measure.label(part, return_num=True, connectivity=2)  # 八邻域

    img_copy = img.copy()
    count = 0
    # measure.regionprops 得到连通区域
    for region2 in measure.regionprops(part_labeled):
        min_row2, min_col2, max_row2, max_col2 = region2.bbox
        w = max_col2-min_col2
        h = max_row2-min_row2
        total_w = maxc-minc
        total_h = maxr-minr
        w_ratio = w/total_w
        h_ratio = h/total_h
        if w_ratio<1/3 and h_ratio<0.2 and w_ratio>0.045 and h_ratio>1/30 and w>=h:
            count = count+1
            img_copy = cv2.rectangle(img_copy, (min_col2 + minc, min_row2 + minr), (max_col2 + minc, max_row2 + minr), (0, 255, 0), 2)

    if count >= 1:
        img = img_copy
        return True, img

    return False, img


def find_face(image_path, kernel_size):
    #读取图像
    image = cv2.imread(image_path)
    #转到ycbcr空间更方便人脸的分离
    image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    height, width, page = image_ycbcr.shape
    y, cb, cr = cv2.split(image_ycbcr)
    # 高斯去噪
    cr_gaussian = cv2.GaussianBlur(cr, (kernel_size, kernel_size), 0)
    cb_gaussian = cv2.GaussianBlur(cb, (kernel_size, kernel_size), 0)

    # 对皮肤的颜色区域变成白色
    skin = np.zeros_like(cr)
    for i in range(height):
        for j in range(width):
            if y[i][j] < 70:
                skin[i][j] = 0
            elif cr_gaussian[i][j] > 133 and cr_gaussian[i][j] < 173 and cb_gaussian[i][j] > 77 and cb_gaussian[i][j] < 127:
                skin[i][j] = 255
            else:
                skin[i][j] = 0

    # 对二值图像形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))   # 得到5x5的十字架
    skin_opening = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)  # 对图像进行开操作

    # 标定连通区域
    skin_labeled = measure.label(skin_opening, connectivity=2)    # 尝试后发现八邻域更好

    count_face = 0
    # 找到连通的脸
    for region in measure.regionprops(skin_labeled):
        min_row, min_col, max_row, max_col = region.bbox
        if (max_row - min_row) / width > 1 / 15 and (max_col - min_col) / height > 0.08: # 参数手动设定
            height_width_ratio = (max_row - min_row) / (max_col - min_col)
            # 比例在(0.6, 2)以内
            if height_width_ratio > 0.6 and height_width_ratio < 2.0:
                # 对可能的脸区域进行眼睛的找寻
                res, image = findeyes(skin_opening, min_row, min_col, max_row, max_col, image)
                if res:
                    count_face = count_face + 1
                    img = cv2.rectangle(image, (min_col, min_row), (max_col, max_row), (0, 0, 255), 2)

    return img

plt.figure()
# img = plt.imread(image_path2)
img = plt.imread(image_path)
plt.subplot(1, 2, 1)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(img)

# img1 = find_face(image_path2, 5)
img1 = find_face(image_path, 5)
plt.subplot(1, 2, 2)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
# 由BGR转换成RGB
img1 = BGR_to_RGB(img1)
plt.imshow(img1)
plt.show()

plt.figure()
img = plt.imread(image_path2)
plt.subplot(1, 2, 1)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(img)

img1 = find_face(image_path2, 5)
plt.subplot(1, 2, 2)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
# 由BGR转换成RGB
img1 = BGR_to_RGB(img1)

plt.imshow(img1)
plt.show()