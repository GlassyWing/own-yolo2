from abc import ABCMeta, abstractmethod

from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Lambda
from keras.layers.merge import concatenate

import tensorflow as tf

FULL_YOLO_BACKEND_PATH = "full_yolo_backend.h5"  # should be hosted on a server


class BaseFeatureExtractor(metaclass=ABCMeta):
    """
    特征提取抽象类
    """

    def __init__(self, input_size):
        input_image = Input((input_size, input_size, 3))
        self.feature_extractor: Model = self.init_feature_extractor(input_image)

    @abstractmethod
    def init_feature_extractor(self, input_image: Input) -> Model:
        raise NotImplementedError("error message")

    @abstractmethod
    def normalize(self, image):
        """
        对输入数据进行标准化
        :param image:
        :return:
        """
        raise NotImplementedError("error message")

    def get_output_shape(self):
        """
        获得输出的特征图的大小
        :return:
        """
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        """
        获得特征图
        :param input_image:
        :return:
        """
        return self.feature_extractor(input_image)


class FullYoloFeature(BaseFeatureExtractor):

    def __init__(self, input_size):
        super().__init__(input_size)
        self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

    def init_feature_extractor(self, input_image) -> Model:
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        def convolutional_block(x, nb_filters: int, stage: int, kernal_size=(3, 3)):
            x = Conv2D(nb_filters, kernel_size=kernal_size, padding='same', name='conv_' + str(stage),
                       use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(stage))(x)
            x = LeakyReLU(alpha=0.1)(x)
            return x

        stage = 1

        # Layer 1
        x = convolutional_block(input_image, 32, stage)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        stage += 1

        # Layer 2
        x = convolutional_block(x, 64, stage)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        stage += 1

        # ==========3x3 => 1x1 => 3x3

        # Layer 3
        x = convolutional_block(x, 128, stage)
        stage += 1

        # Layer 4
        x = convolutional_block(x, 64, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 5
        x = convolutional_block(x, 128, stage)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        stage += 1

        # =========3x3 => 1x1 => 3x3

        # Layer 6
        x = convolutional_block(x, 256, stage)
        stage += 1

        # Layer 7
        x = convolutional_block(x, 128, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 8
        x = convolutional_block(x, 256, stage)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        stage += 1

        # ===========3x3 => 1x1 => 3x3 => 1x1 => 3x3

        # Layer 9
        x = convolutional_block(x, 512, stage)
        stage += 1

        # Layer 10
        x = convolutional_block(x, 256, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 11
        x = convolutional_block(x, 512, stage)
        stage += 1

        # Layer 12
        x = convolutional_block(x, 256, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 13
        x = convolutional_block(x, 512, stage)
        skip_connection = x
        x = MaxPooling2D(pool_size=(2, 2))(x)
        stage += 1

        # ===========3x3 => 1x1 => 3x3 => 1x1 => 3x3

        # Layer 14
        x = convolutional_block(x, 1024, stage)
        stage += 1

        # Layer 15
        x = convolutional_block(x, 512, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 16
        x = convolutional_block(x, 1024, stage)
        stage += 1

        # Layer 17
        x = convolutional_block(x, 512, stage, kernal_size=(1, 1))
        stage += 1

        # Layer 18
        x = convolutional_block(x, 1024, stage)
        stage += 1

        # ===========

        # Layer 19
        x = convolutional_block(x, 1024, stage)
        stage += 1

        # Layer 20
        x = convolutional_block(x, 1024, stage)
        stage += 1

        # Layer 21
        skip_connection = convolutional_block(skip_connection, 64, stage, kernal_size=(1, 1))
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = convolutional_block(x, 1024, stage)

        return Model(input_image, x)

    def normalize(self, image):
        return image / 255
