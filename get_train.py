from keras.callbacks import ModelCheckpoint

import config as c
import get_load
import get_model
import numpy as np
import os
import matplotlib.pyplot as plt

config = c.Config()

train_x, train_y = get_load.get_data(config.train_images_path, config.train_labels_path, config.img_h, config.img_w,
                                     config.num_cls, config.num_chl, config.scale_list)
print('train_x size: ', np.shape(train_x))
print('train_y size: ', np.shape(train_y))
assert (train_y.shape[1] == config.img_h and train_y.shape[2] == config.img_w)

if os.path.isdir(config.model_path):
    pass
else:
    os.mkdir(config.model_path)

model = get_model.upp_model(config.img_h, config.img_w, config.num_chl, config.num_cls, deep_supervision=False)
model_checkpoint = ModelCheckpoint(config.model_path + '/uppnet.hdf5', monitor='loss', save_best_only=True,
                                   save_weights_only=False, verbose=1)
hist = model.fit(x=train_x, y=train_y, batch_size=config.batch_size, epochs=config.num_epoch, verbose=1, shuffle=True,
                 validation_split=float(config.val_rate), callbacks=[model_checkpoint], initial_epoch=0)

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize hist for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
