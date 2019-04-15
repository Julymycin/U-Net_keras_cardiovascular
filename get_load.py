import os
import numpy as np
import cv2
import config as c


def get_data(images_path, labels_path, img_h, img_w, nb_class, nb_channel, scale_list, mode='train'):
    print('loading data')
    assert (nb_class >= 2)
    images = os.listdir(images_path)
    images.sort()
    print(images)
    labels = os.listdir(labels_path)
    labels.sort()
    print(labels)
    shape = []
    for i in range(len(images)):
        image = cv2.imread(os.path.join(images_path, images[i]))
        height, width, _ = image.shape
        shape.append([height, width])
        if i > 0:
            assert shape[i] == shape[i - 1]
    height, width = shape[0]
    stride_w = stride_h = 40
    num_w = 1 + (width - img_w) // stride_w
    num_h = 1 + (height - img_h) // stride_h
    assert (scale_list is not None)
    total_images = np.zeros([len(images) * num_w * num_h, img_h, img_w, nb_channel])  # np.shape:(rows,cols)
    total_labels = np.zeros([len(images) * num_w * num_h, img_h, img_w, nb_class])
    for i in range(len(images)):
        image = cv2.imread(os.path.join(images_path, images[i]))
        label = np.load(os.path.join(labels_path, labels[i]))
        for j in range(num_h):
            for k in range(num_w):
                idx = 36 * i + j * 9 + k
                start_y = 0 + j * stride_h
                end_y = start_y + img_h
                start_x = 0 + k * stride_w
                end_x = start_x + img_w
                total_images[idx, :, :, :] = image[start_y:end_y, start_x:end_x]
                label = label[start_y:end_y, start_x:end_x]
                for cls in range(nb_class):
                    total_labels[idx, :, :, cls] = (label == scale_list[cls]) * 1
    #  mean normalization
    if mode == 'train':
        print('loading train images')
        mean = np.mean(total_images, axis=0)
        np.save('get_load_train_image_mean.npy', mean)  # for test data normalization
    elif mode == 'test':
        print('loading test images')
        mean = np.load('get_load_train_image_mean.npy')
    total_images = (total_images - mean) / (np.max(total_images) - np.min(total_images))

    return total_images, total_labels


if __name__ == '__main__':
    config = c.Config()
    x, y = get_data(config.train_images_path, config.train_labels_path, config.img_h, config.img_w,
                    config.num_cls, config.num_chl, config.scale_list)
