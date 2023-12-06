import cv2
import os
import numpy as np
import random

class ImagePre:
    def __init__(self, input_pic, resize_size=None, crop_size=None, flip_code=None, ksize =None, resize_interpolation=None):
        self.input_pic = input_pic
        self.data_pic = self.load_images()
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_code = flip_code
        self.resize_interpolation = resize_interpolation
        self.ksize = ksize

    # 导入图片
    def load_images(self):
        data_pic = []
        for filename in os.listdir(self.input_pic):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(self.input_pic, filename)
                img = cv2.imread(img_path)
                data_pic.append(img)
        return data_pic

    # 改变图片大小(替换原图片)
    def resize_image(self, interpolation=cv2.INTER_LINEAR):
        for i, img in enumerate(self.data_pic):
            self.data_pic[i] = cv2.resize(img, self.resize_size, interpolation=self.resize_interpolation)

    # 随机剪裁图片
    def crop(self):
        cropped_data_pic = []
        for img in self.data_pic:
            h, w, _ = img.shape
            left = random.randint(0, w - self.crop_size[0])
            top = random.randint(0, h - self.crop_size[1])
            cropped_data_pic.append(img[top:top + self.crop_size[1], left:left + self.crop_size[0]])
        self.cropped_data_pic = cropped_data_pic

    # 翻转图片
    def flip(self):
        flipped_data_pic = [cv2.flip(img, self.flip_code) for img in self.data_pic]
        self.flipped_data_pic = flipped_data_pic

    # 灰度化图片并提高对比度(替换原图片)
    def Grayscale_Histogram_Equalization(self):
        for i, img in enumerate(self.data_pic):
            self.data_pic[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i, img in enumerate(self.data_pic):
            self.data_pic[i] = cv2.equalizeHist(img)

    # 去除图片噪声(替换图片)
    def Noise_Removal(self):
        for i, img in enumerate(self.data_pic):
            self.data_pic[i] = cv2.medianBlur(img, self.ksize)

    # 随机在角度(-45,45)间旋转
    def random_rotation(self, angle_range=(-45, 45), scale=1.0):
        random_rotation_data_pic = []
        # 获取随机角度
        angle = np.random.uniform(angle_range[0], angle_range[1])
        for img in self.data_pic:
            # 获取图像中心点坐标
            height, width = img.shape[:2]
            center = (width / 2, height / 2)
            # 构建旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            # 进行旋转,并添加到random_rotation_data_pic = []列表
            random_rotation_data_pic.append(cv2.warpAffine(img, rotation_matrix, (width, height)))
        self.random_rotation_data_pic = random_rotation_data_pic

    # 图像零均值化(替换图片)
    def zero_mean(self):
        for i, img in enumerate(self.data_pic):
            mean_value = np.mean(img, axis=(0, 1))  # 计算每个通道的均值
            zero_mean_img = img - mean_value
            self.data_pic[i] = zero_mean_img

    # 图片归一化(替换图片)
    def std(self):
        for i, img in enumerate(self.data_pic):
            std_value = np.std(img, axis=(0, 1))  # 计算每个通道的标准差
            normalized_img = img / std_value
            self.data_pic[i] = normalized_img

    def append(self):
        self.data_pic.extend(self.cropped_data_pic)
        self.data_pic.extend(self.flipped_data_pic)
        self.data_pic.extend(self.random_rotation_data_pic)


input_pic = r"C:\Users\86185\Desktop\1111"
image_processor = ImagePre(
    input_pic=input_pic,
    resize_size=(100, 100),  # 你可以根据需要调整大小
    crop_size=(80, 80),  # 你可以根据需要进行裁剪
    flip_code=1,  # 1 表示水平翻转，0 表示垂直翻转，-1 表示水平和垂直翻转
    resize_interpolation=cv2.INTER_NEAREST  # 你可以选择其他的插值方法
)
image_processor.resize_image()
image_processor.crop()
image_processor.flip()
image_processor.append()
datas = image_processor.data_pic
print(len(datas))