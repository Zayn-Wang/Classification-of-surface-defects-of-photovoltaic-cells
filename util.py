from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from ImageDataAugmentor.image_data_augmentor import *
import albumentations

# way1 0.7710
AUGMENTATIONS = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
    ],p=1),
    albumentations.GaussianBlur(p=0.05),
    albumentations.OneOf([
            albumentations.MotionBlur(p=0.2),
            albumentations.MedianBlur(blur_limit=3, p=0.1),
            albumentations.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
    albumentations.HueSaturationValue(p=0.5),
    albumentations.RGBShift(p=0.5),
])

# way3
# AUGMENTATIONS = albumentations.Compose([
#         albumentations.RandomRotate90(),
#         albumentations.Flip(),
#         albumentations.Transpose(),
#         albumentations.OneOf([
#             albumentations.IAAAdditiveGaussianNoise(),
#             albumentations.GaussNoise(),
#         ], p=0.2),
#         albumentations.OneOf([
#             albumentations.MotionBlur(p=0.2),
#             albumentations.MedianBlur(blur_limit=3, p=0.1),
#             albumentations.Blur(blur_limit=3, p=0.1),
#         ], p=0.2),
#         #albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
#         albumentations.OneOf([
#             albumentations.OpticalDistortion(p=0.3),
#             albumentations.GridDistortion(p=0.1),
#             albumentations.IAAPiecewiseAffine(p=0.3),
#         ], p=0.2),
#         albumentations.OneOf([
#             albumentations.CLAHE(clip_limit=2),
#             albumentations.IAASharpen(),
#             albumentations.IAAEmboss(),
#             albumentations.RandomBrightnessContrast(),
#         ], p=0.3),
#         albumentations.HueSaturationValue(p=0.3),
#     ], p=0.5)

def get_data_newfour(batch_size=32):
    data_dir = 'train_data_all'
    img_height = 300
    img_width = 300
    validation_dir = 'test_data_all'

    datagen = ImageDataGenerator(
        rescale=1. / 255)  # 图像像素的归一化
        #rotation_range=40,  # 通过旋转图像的角度扩大训练集
        # width_shift_range=0.2,  # 通过水平偏移图像来扩大训练集
        # height_shift_range=0.2,  # 通过垂直偏移图像来扩大训练集
        # shear_range=0.2,  # 通过随机裁剪的角度来扩大训练集
        #zoom_range=0.2,  # 通过随机缩放来扩大训练集
        # horizontal_flip=True,  # 通过随机将一半图像水平翻转来扩大训练
        # vertical_flip=True)
        #fill_mode='nearest')  # 部分操作过后，产生的像素缺失，通过填充最近的像素来补全
    # datagen = ImageDataAugmentor(
    #     rescale=1./255,
    #     augment=AUGMENTATIONS,
    #     preprocess_input=None)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_ds = datagen.flow_from_directory(
        data_dir,
        # subset="training",

        class_mode='categorical',
        target_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = test_datagen.flow_from_directory(
        validation_dir,
        # validation_split=0.2,
        # subset="validation",

        class_mode='categorical',
        target_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds, val_ds

