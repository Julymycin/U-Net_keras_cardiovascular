import numpy as np


class Config:
    def __init__(self):
        # data path
        self.train_images_path = './train_images/'
        self.train_labels_path = './train_labels/'
        self.test_images_path = './test_images/'
        self.test_labels_path = './test_labels/'
        self.model_dir = './model/'
        self.source_images_path = './source_images/'
        self.source_labels_path = './source_labels/'
        self.predict_images_path = './pred_images/'

        # data parameters
        self.img_h = 256
        self.img_w = 256
        self.num_cls = 3
        self.num_chl = 3
        self.scale_list = np.arange(self.num_cls)

        # train parameters
        self.num_epoch = 30
        self.batch_size = 8
        self.val_rate = 0.1
