from __future__ import print_function

from model_unet import *
from model_segnet import *
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"
data_path = '/data2/yeom/ky_fetal/cs230_data/train/axnew/'
nbatch = 4

x_tr = np.load(data_path+'x_tr.npy')
y_tr = np.load(data_path+'y_tr.npy')
x_dev = np.load(data_path+'x_dev.npy')
y_dev = np.load(data_path+'y_dev.npy')

model = unet()
#model = segnet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

training_loss=np.zeros((100, 1))
test_loss=np.zeros((100, 1))
minloss = 100
flag = 0

history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
model.save('model_trained.h5')

training_loss[0] = history.history['loss']
test_loss[0] = history.history['val_loss']
if test_loss[0]<minloss:
    minloss = test_loss[0]
    model.save('model_best.h5')

for i in range(99):
    #model = load_model('model_trained.h5', custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D': MaxUnpooling2D})
    model = load_model('model_trained.h5')
    history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=1, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
    os.remove('model_trained.h5')
    model.save('model_trained.h5')
    training_loss[i+1] = history.history['loss']
    test_loss[i+1] = history.history['val_loss']
    if test_loss[i+1]<minloss:
        flag = 0
        minloss = test_loss[i+1]
        os.remove('model_best.h5')
        model.save('model_best.h5')
    else:
        flag = flag + 1

    if flag == 5:
        break

np.savetxt('training_loss.txt', training_loss, delimiter = " ", fmt = "%s")
np.savetxt('test_loss.txt', test_loss, delimiter = " ", fmt = "%s")

