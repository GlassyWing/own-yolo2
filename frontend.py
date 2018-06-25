import numpy as np
from keras.layers import Input, Conv2D, Reshape, Lambda
from keras import Model

from backend import FullYoloFeature


class YOLO:

    def __init__(self, backend,
                 input_size,
                 labels,
                 max_box_per_image,
                 anchors):
        """

        :param backend: 特征提取器
        :param input_size: 输入图像的维度
        :param labels: 标签
        :param max_box_per_image: 每张图像所拥有的最多框
        :param anchors: 锚框
        """

        self.input_size = input_size
        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box = len(anchors) // 2
        self.class_wt = np.ones(self.nb_class, dtype=np.float32)
        self.anchors = anchors
        self.max_box_per_image = max_box_per_image

        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

        self.feature_extractor = FullYoloFeature(self.input_size)
        print(self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # 创建物体检测层
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)
