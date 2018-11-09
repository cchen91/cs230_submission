from __future__ import print_function

from model import *

os.environ["CUDA_VISIBLE_DEVICES"]="3"
data_path = '/data2/yeom/ky_fetal/cs230_data/train/ax/'
nbatch = 16

x_tr = np.load(data_path+'x_tr.npy')
y_tr = np.load(data_path+'y_tr.npy')
x_dev = np.load(data_path+'x_dev.npy')
y_dev = np.load(data_path+'y_dev.npy')

model = unet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=100, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
model.save('model_trained.h5')
training_loss = history.history['loss']
test_loss = history.history['val_loss']
np.savetxt('training_loss.txt', training_loss, delimiter = " ", fmt = "%s")
np.savetxt('test_loss.txt', test_loss, delimiter = " ", fmt = "%s")

