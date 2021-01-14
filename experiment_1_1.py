import matplotlib.pyplot as plt
# 1-1 显示不同类型的图像
image_path1 = './images/pro_1/'
im = plt.imread(image_path1+'Img1.png')
im2 = plt.imread(image_path1+'Img2.jpg')
im3 = plt.imread(image_path1+'Img3.bmp')
im4 = plt.imread(image_path1+'Img4.gif')
plt.subplot(2, 2, 1)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(im)
plt.title("Img1.png")

plt.subplot(2, 2, 2)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(im2)
plt.title("Img2.jpg")

plt.subplot(2, 2, 3)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(im3)
plt.title("Img3.bmp")

plt.subplot(2, 2, 4)
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
plt.imshow(im4)
plt.title("Img4.gif")
plt.show()



