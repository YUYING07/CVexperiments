import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_disparity(left_path, right_path, n, m, dmax=30, alpha=1):
    '''

    :param left_path:  左视图
    :param right_path: 右视图
    :param n:          窗口的行
    :param m:          窗口的列
    :param dmax:       最大迭代次数
    :param alpha:      阈值权重
    :return:
    '''
    # 读取得到左图 和 右图
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    # 得到图像的长宽
    length, width = left.shape[0], left.shape[1]
    # 滤波窗口的大小

    # 得到e_avg和最基本的视差图
    disparity = np.zeros((length, width), dtype=np.uint8)

    # error_energy
    e = np.zeros((length, width, dmax), dtype=np.float)
    e_avg = np.zeros((length, width, dmax), dtype=np.float)

    # 计算得到 error energy matrix

    # 由于元素 在 边缘像素肯定会因此越界 准备padding
    imgLeft = cv2.copyMakeBorder(left, 0, n-1, 0, m - 1 + dmax, borderType=cv2.BORDER_REPLICATE) # 图片下侧
    imgRight = cv2.copyMakeBorder(right, 0, n-1, 0, m - 1, borderType=cv2.BORDER_REPLICATE)  # 图片右侧
    # 迭代 dmax 次
    for d in range(dmax):
        # 对整个图像进行遍历
        for i in range(length):
            for j in range(width):
                # 对于每个像素的(i, j, d) 根据公式(1)计算误差能量矩阵
                errorEnergy = (imgLeft[i:i+n, j + d:j + m + d, :] - imgRight[i:i + n, j:j + m, :]) ** 2

                # 对每个维度进行求和
                errorEnergy = errorEnergy[:, :, 0]+errorEnergy[:, :, 1]+errorEnergy[:, :, 2]
                e[i, j, d] = np.sum(errorEnergy) / (3 * n * m)

        # 再对d情况下的e进行平均滤波得到e_avg(i,j,d)
        for i in range(length):
            for j in range(width):
                e_avg[i, j, d] = np.sum(e[i:i+n, j:j+m, d])/(n*m)

        # 按照论文进行多次均值滤波去噪声
        for k in range(5):
            for i in range(length):
                for j in range(width):
                    e_avg[i, j, d] = np.sum(e_avg[i:i + n, j:j + m, d]) / (n * m)

    # 取到最小d值 映射到 disparity
    disparity[:, :] = np.argmin(e_avg, axis=2)  # 视差图
    res_disparity = disparity.copy()

    # 计算具有可靠视差的视差图
    # 得到决定视图预测可靠与否的误差能量阈值 和论文中的式子一致
    Ve = alpha * np.mean(e_avg)
    count_not_ne = 0
    sum_e = 0.0
    for i in range(length):
        for j in range(width):
            if e_avg[i][j][disparity[i][j]] > Ve:                  # 此时能量误差不可估计（ne）
                disparity[i][j] = 0               # 得到更可靠的视差
            else:
                sum_e = sum_e + e_avg[i][j][disparity[i][j]]       # 计算可靠视差的能量误差
                count_not_ne = count_not_ne + 1   # 此时 not ne

    # 根据可信度公式 计算可信度 Rd
    reliability = float(count_not_ne) ** (-1) * (sum_e ** (-1))
    print("可信度：", reliability)

    return res_disparity, disparity

def get_depth(disparity, f, T):

    length, width = disparity.shape
    # 保证深度图的光滑 要先用 5 x 5的窗口进行中值滤波
    cv2.medianBlur(disparity, 5)

    # 计算深度图，并显示3D视图
    depth = np.ones_like(disparity, dtype=np.uint8)
    for i in range(length):
        for j in range(width):
            if disparity[i][j] == 0:
                depth[i][j] = 0
            else:
                depth[i][j] = f * T // disparity[i][j]

    return depth


if __name__ == '__main__':

    dmax = 40  # 搜寻匹配的范围
    m = 3
    n = 3
    f = 30  # 焦距
    T = 20  # 两个相机间距
    alpha = 1  # 阈值ve的系数
    left_path = './images/pro_5/view1m.png'
    right_path = './images/pro_5/view5m.png'
    # 展示左右两张图
    plt.subplot(1, 2, 1)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.imshow(plt.imread(left_path))
    plt.subplot(1, 2, 2)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.imshow(plt.imread(right_path))
    plt.show()

    # 展示视差图
    res_disparity, disparity = get_disparity(left_path, right_path, m, n, dmax, alpha)
    plt.figure()
    plt.imshow(res_disparity, cmap='gray', vmin=0, vmax=40)
    plt.title('disparity figure')  # 设置标题
    ax = plt.gca()  # 返回坐标轴实例
    x_major_locator = plt.MultipleLocator(20)  # 刻度间隔为20
    y_major_locator = plt.MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)  # 设置坐标间隔
    ax.yaxis.set_major_locator(y_major_locator)
    plt.colorbar()  # 设置colorBar
    plt.show()

    # 展示深度图
    depth = get_depth(disparity, f, T)
    # 深度图图像
    plt.figure()
    plt.imshow(depth, cmap='gray', vmin=0, vmax=120)
    plt.title('depth figure')  # 设置标题
    ax = plt.gca()  # 返回坐标轴实例
    x_major_locator = plt.MultipleLocator(20)  # 刻度间隔为20
    y_major_locator = plt.MultipleLocator(20)
    ax.xaxis.set_major_locator(x_major_locator)  # 设置坐标间隔
    ax.yaxis.set_major_locator(y_major_locator)
    plt.colorbar()  # 设置colorBar
    plt.show()

    # 三维视图
    fig = plt.figure()
    ax = Axes3D(fig)
    rows, cols = depth.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    z = depth
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                           vmin=0, vmax=40)
    ax.set_title('3D figure')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
