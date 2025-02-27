import cv2
import numpy as np
from PIL import Image
import os
from skimage.filters import gabor
from scipy.stats import entropy


def is_snow_screen(image_path, threshold=0.001, variance_threshold=1000, entropy_threshold=1.5):
    # 确保路径是正确的格式
    image_path = os.path.normpath(image_path)

    # 使用PIL读取图片
    try:
        with Image.open(image_path) as img:
            # 将PIL图像转换为OpenCV格式（灰度图像）
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    except Exception as e:
        raise ValueError(f"无法读取图片: {image_path}, 错误: {e}")

    # 应用中值滤波来减少噪点
    filtered_image = cv2.medianBlur(image, 3)

    # 计算原始图像和滤波后的图像的差值
    noise_image = cv2.absdiff(image, filtered_image)

    # 计算图片的总像素数
    total_pixels = image.size

    # 计算噪点像素的数量
    noise_pixels = np.sum(noise_image > 25)

    # 计算噪点密度
    noise_density = noise_pixels / total_pixels

    # 计算噪点像素的位置
    noise_positions = np.argwhere(noise_image > 25)

    # 计算噪点位置的方差
    if noise_positions.size > 0:
        variance = np.var(noise_positions, axis=0).mean()
    else:
        variance = 0

    # 计算噪点图像的Shannon熵
    hist, _ = np.histogram(noise_image, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    entropy_value = entropy(hist)

    # 使用Gabor滤波器计算响应
    gabor_real, gabor_imag = gabor(noise_image, frequency=0.1)
    gabor_response = np.sqrt(gabor_real ** 2 + gabor_imag ** 2)
    gabor_mean = gabor_response.mean()

    # 如果噪点密度超过设定的阈值且方差超过设定的阈值且熵超过设定的阈值且Gabor响应均值超过设定的阈值，则认为是雪花屏
    return (noise_density > threshold and
            variance > variance_threshold and
            entropy_value > entropy_threshold and
            gabor_mean > 0.1)


# 使用示例
image_path = r'C:\Users\Administrator\Desktop\雪花屏样本png.png'
if is_snow_screen(image_path):
    print("该图片是雪花屏")
else:
    print("该图片不是雪花屏")
