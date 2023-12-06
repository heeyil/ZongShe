import cv2
import os
import random
import matplotlib.pyplot as plt


class ImagePre:
    def __init__(self, input_pic, resize_size=None, crop_size=None, flip_code=None, resize_interpolation=None):
        self.input_pic = input_pic
        self.data_pic = self.load_images()
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_code = flip_code
        self.resize_interpolation = resize_interpolation

    def load_images(self):
        data_pic = []
        for filename in os.listdir(self.input_pic):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(self.input_pic, filename)
                img = cv2.imread(img_path)
                data_pic.append(img)
        return data_pic

    def resize_image(self, interpolation=cv2.INTER_LINEAR):
        for i, img in enumerate(self.data_pic):
            self.data_pic[i] = cv2.resize(img, self.resize_size, interpolation=self.resize_interpolation)

    def crop(self):
        cropped_data_pic = []
        for img in self.data_pic:
            h, w, _ = img.shape
            left = random.randint(0, w - self.crop_size[0])
            top = random.randint(0, h - self.crop_size[1])
            cropped_data_pic.append(img[top:top + self.crop_size[1], left:left + self.crop_size[0]])
        self.cropped_data_pic = cropped_data_pic

    def flip(self):
        flipped_data_pic = [cv2.flip(img, self.flip_code) for img in self.data_pic]
        self.flipped_data_pic = flipped_data_pic

    def append(self):
        self.data_pic.extend(self.cropped_data_pic)
        self.data_pic.extend(self.flipped_data_pic)


input_pic = r"D:\test_picture"
image_processor = ImagePre(
    input_pic=input_pic,
    resize_size=(200, 200),  # 你可以根据需要调整大小
    crop_size=(80, 80),  # 你可以根据需要进行裁剪
    flip_code=1,  # 1 表示水平翻转，0 表示垂直翻转，-1 表示水平和垂直翻转
    resize_interpolation=cv2.Bicubic_Interpolation  # 你可以选择其他的插值方法
)
image_processor.resize_image()
image_processor.crop()
image_processor.flip()
image_processor.append()
datas = image_processor.data_pic
print(len(datas))


def show_images(images):
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i+1}")

    plt.show()


# 显示图像列表
show_images(datas)

