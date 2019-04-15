import config as c
from keras.models import load_model
import os
import get_load

config = c.Config()
model = load_model(os.path.join(config.model_path, 'uppnet.hdf5'))
test_x, test_y = get_load.get_data(config.test_images_path, config.test_labels_path, config.img_h, config.img_w,
                                   config.num_cls, config.num_chl, config.scale_list, mode='test')
loss, accuracy = model.evaluate(test_x, test_y, batch_size=config.batch_size, verbose=1)
print("loss: %.2f, acc: %.2f%%" % (loss, accuracy * 100))
