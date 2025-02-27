import cv2
import numpy as np

def detect_image_properties(image_path: str) -> None:
    """
    判断图片的清晰度、是否是黑白图片、是否是雪花图、是否过亮或过暗，以及是否偏色、黑屏或树叶遮挡，并直接输出中文结果。

    参数:
        image_path (str): 图片路径
    """
    # 读取图片
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("无法加载图片，请检查路径是否正确")

    # 检查图片通道数
    is_gray_image = len(image.shape) == 2  # 单通道（灰度图）
    is_bgra_image = len(image.shape) == 3 and image.shape[2] == 4  # 4 通道（BGRA 图像）

    # 如果是 4 通道（BGRA），忽略 Alpha 通道
    if is_bgra_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # 1. 判断是否是黑白图片
    if is_gray_image:  # 单通道
        is_black_white = True
    else:
        b, g, r = cv2.split(image)
        if np.allclose(b, g, atol=5) and np.allclose(g, r, atol=5):  # 判断是否是伪黑白图
            is_black_white = True
        else:
            is_black_white = False
    print(f"黑白检测：{'是黑白图片' if is_black_white else '不是黑白图片'}")

    # 2. 判断图片清晰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if not is_gray_image else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clear = laplacian_var > 100  # 阈值可以调整，低于100认为模糊
    print(f"清晰度检测：{'清晰' if is_clear else '模糊'}，清晰度评分为 {laplacian_var:.2f}")

    # 3. 判断是否是雪花图
    noise = cv2.meanStdDev(gray)[1][0][0]  # 计算标准差
    is_snowy = noise > 50  # 阈值可以调整，标准差过高认为是雪花图
    print(f"雪花检测：{'是雪花图' if is_snowy else '不是雪花图'}，噪声水平为 {noise:.2f}")

    # 4. 判断是否过亮或过暗
    brightness = np.mean(gray)  # 计算亮度
    if brightness < 50:
        brightness_status = "过暗"
    elif brightness > 200:
        brightness_status = "过亮"
    else:
        brightness_status = "正常"
    print(f"亮度检测：{brightness_status}，平均亮度为 {brightness:.2f}")

    # 5. 偏色检测（仅对彩色图像进行）
    if not is_gray_image:
        total_pixels = image.shape[0] * image.shape[1]
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        b_ratio, g_ratio, r_ratio = b_mean / 255, g_mean / 255, r_mean / 255  # 转为比例
        max_ratio = max(b_ratio, g_ratio, r_ratio)
        min_ratio = min(b_ratio, g_ratio, r_ratio)

        # 判断偏色
        if max_ratio - min_ratio > 0.2:  # 偏色阈值，0.2 表示 20% 偏差
            if max_ratio == r_ratio:
                color_bias = "偏红"
            elif max_ratio == g_ratio:
                color_bias = "偏绿"
            elif max_ratio == b_ratio:
                color_bias = "偏蓝"
        else:
            color_bias = "无偏色"
        print(f"偏色检测：{color_bias}（蓝色平均值：{b_mean:.2f}，绿色平均值：{g_mean:.2f}，红色平均值：{r_mean:.2f}）")
    else:
        print("偏色检测：无法检测（灰度图像）")

    # 6. 黑屏检测
    is_black_screen = brightness < 10  # 平均亮度过低，判断为黑屏
    print(f"黑屏检测：{'是黑屏图片' if is_black_screen else '不是黑屏图片'}")

    # 7. 树叶遮挡检测（仅对彩色图像进行）
    if not is_gray_image:
        if is_snowy:
            # 如果是雪花图，直接跳过树叶遮挡检测
            print("树叶遮挡检测：无法检测（雪花图）")
        else:
            green_mask = (g > r + 30) & (g > b + 30)  # 绿色通道显著高于其他通道
            green_ratio = np.sum(green_mask) / (image.shape[0] * image.shape[1])  # 绿色像素比例

            # 检测绿色区域的连通性
            green_area = np.uint8(green_mask * 255)  # 转为二值图像
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_area)
            max_component_area = max(stats[1:, cv2.CC_STAT_AREA], default=0)  # 最大连通区域面积

            # 判断是否树叶遮挡
            is_leaf_blocked = green_ratio > 0.4 and max_component_area > 500  # 连通面积阈值 500
            print(f"树叶遮挡检测：{'是树叶遮挡' if is_leaf_blocked else '不是树叶遮挡'}，绿色像素比例为 {green_ratio:.2%}，最大绿色连通区域面积为 {max_component_area}")
    else:
        print("树叶遮挡检测：无法检测（灰度图像）")


# 示例使用
#image_path = r"C:\Users\Administrator\Desktop/1.png"  # 替换为你的图片路径
#image_path = r"D:\样本\Snow/snow_storm-147.jpg"  # 替换为你的图片路径
image_path = r"D:\样本\Rain\rain_storm-622.jpg"  # 替换为你的图片路径
detect_image_properties(image_path)
