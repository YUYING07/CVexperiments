import math
import numpy as np
import matplotlib.pyplot as plt

def Gaussian_kernel(sigma):

    #滤波窗口大小 [6*sigma-1]/2*2+1
    Kernel_size = math.floor(6*sigma-1)//2*2+1
    #生成高斯滤波器的核
    half = Kernel_size//2
    Kernel = np.zeros((Kernel_size, Kernel_size), dtype = np.float)
    for x in range(-half, -half+Kernel_size):
        for y in range(-half, -half+Kernel_size):
            # 不能是负的索引
            Kernel[x+half, y+half] = np.exp(-(x**2+y**2)/(2*(sigma**2)))
    Kernel /= (2*np.pi*sigma**2)
    Kernel /= Kernel.sum()
    return Kernel

    #进行高斯滤波
def Gaussian_kernel_onedim(sigma):

    #滤波窗口大小 [6*sigma-1]/2*2+1
    Kernel_size = math.floor(6*sigma-1)//2*2+1
    #生成高斯滤波器的核
    half = Kernel_size//2
    Kernel = np.zeros((Kernel_size), dtype=np.float)
    for x in range(-half, -half+Kernel_size):
            # 不能是负的索引
            Kernel[x+half] = np.exp(-(x**2)/(2*(sigma**2)))
    Kernel /= np.sqrt(2*np.pi)*sigma
    Kernel /= Kernel.sum()
    # 验证卷积核的正确性
    # Kernel = Kernel.reshape(Kernel.shape[0], 1)
    # print(np.dot(Kernel, Kernel.T))
    return Kernel, Kernel_size
    #进行高斯滤波

def Guass(sigma,image):
    # print(Gaussian_kernel_onedim(sigma))
    kernel, kernel_size = Gaussian_kernel_onedim(sigma)
    res_img_size_x = image.shape[0]-kernel_size//2*2
    res_image = np.zeros((res_img_size_x, image.shape[1], 3), dtype=np.float)
    # 先对行进行 一维卷积
    for i in range(res_img_size_x):
        for j in range(image.shape[1]):
            sum = 0
            for k in range(kernel_size):
                sum = sum+image[i+k][j]*kernel[k]
            res_image[i][j] = sum

    #对中间结果的列 进行一维卷积
    res_img_size_y = image.shape[1] - kernel_size // 2 * 2
    final_res_image = np.zeros((res_img_size_x, res_img_size_y, 3), dtype=np.float)
    for j in range(res_img_size_y):
        for i in range(res_img_size_x):
            sum= 0
            for k in range(kernel_size):
                sum= sum+res_image[i][j+k]*kernel[k]
            final_res_image[i][j] = sum
    # 0-1之间显示 故要/255
    final_res_image /= 255
    plt.subplot(1, 2, 2)
    plt.imshow(final_res_image)
    plt.show()

Gaussian_kernel(1)
Gaussian_kernel_onedim(1)
image_path1 = './images/pro_2/a.jpg'
img = plt.imread(image_path1)
plt.subplot(1, 2, 1)
plt.imshow(img)

Guass(1, img)

