import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_kernel(x, y, sigame):  # 由于高斯分布在x和y方向上的导数仅有乘以x还是y的区别，故只需计算一个方向
    return (-1 / (2 * np.pi * sigame ** 4)) * x * np.exp(-(x ** 2 + y ** 2) / (2 * sigame ** 2))

def get_kernel_y(x, y, sigame):
    return (-1 / (2 * np.pi * sigame ** 4)) * y * np.exp(-(x ** 2 + y ** 2) / (2 * sigame ** 2))

def my_gray(img):  # 得到灰度图
    temp = np.array([0.299, 0.587, 0.114])  # 灰度图像计算公式R * 0.299 + G * 0.587 + B * 0.114
    return img @ temp

def my_Gaussian(img, sigame):
    # 计算高斯分布x方向和y方向的一阶差分
    kernel_x = np.zeros((3, 3))
    kernel_y = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            kernel_x[i][j] = get_kernel(i - 1, j - 1, sigame)
            kernel_y[i][j] = get_kernel(j - 1, i - 1, sigame)
    # 对原图像进行补零，使得卷积后的图像与原图像大小相等
    img_padding = np.pad(img, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)
    # 与图像卷积
    angle = np.zeros_like(img)
    result_x = np.zeros_like(img)
    result_y = np.zeros_like(img)
    result = np.zeros_like(img)
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            area = img_padding[i :i + 3, j:j + 3]  # 得到卷积时图像中3 * 3的区域
            result_x[i, j] = np.sum(area * kernel_x)
            result_y[i, j] = np.sum(area * kernel_y)
            result[i, j] = (result_x[i][j] ** 2 + result_y[i][j] ** 2) ** 0.5

            if result_x[i, j] > 0 and result_y[i, j] > 0:
                angle[i, j] = np.arctan(result_y[i, j] / result_x[i, j])

            if result_x[i, j] < 0 and result_y[i, j] > 0:
                angle[i, j] = np.arctan(result_y[i, j] / result_x[i, j]) + np.pi

            if result_x[i, j] < 0 and result_y[i, j] < 0:
                angle[i, j] = np.arctan(result_y[i, j] / result_x[i, j]) + np.pi
            if result_x[i, j] > 0 and result_y[i, j] < 0:
                angle[i, j] = np.arctan(result_y[i, j] / result_x[i, j])
            if result_x[i, j] == 0 and result_y[i, j] > 0:
                angle[i, j] = np.pi / 2
            if result_x[i, j] == 0 and result_y[i, j] < 0:
                angle[i, j] = -np.pi / 2
    # -π/8 ~ π/8 和 7π/8 ~ 9π/8范围内角度设置为 0
    # π/8 ~ 3π/8 和 9π/8 ~ 11π/8范围内角度设置为 π/4
    # 5π/8 ~ 7π/8 和 -3π/8 ~ -π/8范围内角度设置为 3π/4
    # 其余设置为 π/2
    for i in range(angle.shape[0]):
        for j in range(angle.shape[1]):
            if (angle[i][j] >= -np.pi / 8 and angle[i][j] < np.pi / 8) or (angle[i][j] >= 7 * np.pi / 8 and angle[i][j] < 9 * np.pi / 8):
                angle[i][j] = 0
            elif (angle[i][j] >= np.pi / 8 and angle[i][j] < 3 * np.pi / 8) or (angle[i][j] >= 9 * np.pi / 8 and angle[i][j] < 11 * np.pi / 8):
                angle[i][j] = np.pi / 4
            elif (angle[i][j] >= 5 * np.pi / 8 and angle[i][j] < 7 * np.pi / 8) or (angle[i][j] >= -3 * np.pi / 8 and angle[i][j] < -np.pi / 8):
                angle[i][j] = 3 * np.pi / 4
            else:
                angle[i][j] = np.pi / 2
    return result, angle

def my_NMS(img, angle):  # 对图像进行非极大值抑制
    result = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1:
                continue
            if angle[i][j] == 0:
                if img[i][j] < img[i - 1][j] or img[i][j] < img[i + 1][j]:
                    result[i][j] = 0
            if angle[i][j] == np.pi / 4:
                if img[i][j] < img[i + 1][j + 1] or img[i][j] < img[i - 1][j - 1]:
                    result[i][j] = 0

            if angle[i][j] == np.pi / 2:
                if img[i][j] < img[i][j - 1] or img[i][j] < img[i][j + 1]:
                    result[i][j] = 0
            if angle[i][j] == 3 * np.pi / 4:
                if img[i][j] < img[i - 1][j + 1] or img[i][j] < img[i + 1][j - 1]:
                    result[i][j] = 0
    return result

def my_threhold(img, lower=14, upper=20):
    result = img.copy()
    # 根据阈值确定边缘
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <= lower:
                result[i][j] = 0
            if img[i][j] >= upper:
                result[i][j] = 255
    # 对于大于lower小于upper的像素点，如果它上下或者左右或者左上角右下角或者左下角右上角中四种情况有一种都是边缘，则该像素点也是边缘
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i == 0 or i == img.shape[0] - 1 or j == 0 or j == img.shape[1] - 1:
                continue
            if result[i][j] == 0 or result[i][j] == 255:
                continue

            if result[i - 1][j] == 255 and result[i + 1][j] == 255:
                result[i][j] = 255
            elif result[i][j - 1] == 255 and result[i][j + 1] == 255:
                result[i][j] = 255
            elif result[i + 1][j + 1] == 255 and result[i - 1][j - 1] == 255:
                result[i][j] = 255
            elif result[i + 1][j - 1] == 255 and result[i - 1][j + 1] == 255:
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result

img = Image.open("1.jpg")
img = np.array(img)
gray_img = my_gray(img)
img_G, angle = my_Gaussian(gray_img, 1)

img_NMS = my_NMS(img_G, angle)

result = my_threhold(img_NMS, 10, 20)

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
plt.imshow(img_G, cmap="gray")
ax1.set_title("gray")
ax2 = fig.add_subplot(1, 3, 2)
plt.imshow(img_NMS, cmap="gray")
ax2.set_title("NMS")
ax3 = fig.add_subplot(1, 3, 3)
plt.imshow(result, cmap="gray")
ax3.set_title("threhold")
plt.show()
