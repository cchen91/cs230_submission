from __future__ import print_function

from model_unet import *
from model_segnet import *
import matplotlib.image as mpimg

os.environ["CUDA_VISIBLE_DEVICES"]="3"
data_path = '/data2/yeom/ky_fetal/cs230_data/train/axnew/'

nbatch = 16

x_tr = np.load(data_path+'x_tr.npy')
y_tr = np.load(data_path+'y_tr.npy')
x_dev = np.load(data_path+'x_dev.npy')
y_dev = np.load(data_path+'y_dev.npy')
x_te = np.load(data_path+'x_te.npy')
y_te = np.load(data_path+'y_te.npy')

model = unet()
#model = segnet()
#model = load_model('model_trained.h5', custom_objects={'iou_coef_loss': iou_coef_loss, 'iou_coef': iou_coef})
#model = load_model('model_trained.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
model = load_model('model_trained.h5')

yhat_tr = model.predict(x_tr, batch_size = nbatch, verbose=1)
yhat_tr = np.floor(yhat_tr+0.5)
yhat_dev = model.predict(x_dev, batch_size = nbatch, verbose=1)
yhat_dev = np.floor(yhat_dev+0.5)
yhat_te = model.predict(x_te, batch_size = nbatch, verbose=1)
yhat_te = np.floor(yhat_te+0.5)

y_tr.astype(bool)
yhat_tr.astype(bool)
y_dev.astype(bool)
yhat_dev.astype(bool)
y_te.astype(bool)
yhat_te.astype(bool)

# Evaluating IoU score for training set
inter_tr = np.sum(np.logical_and(y_tr, yhat_tr), axis = (1, 2))+K.epsilon()
union_tr = np.sum(np.logical_or(y_tr, yhat_tr), axis = (1, 2))+K.epsilon()
iou_tr = inter_tr/union_tr
print('\n\r IoU score mean of training set:\n\r')
print(iou_tr.mean())

# Evaluating IoU score for dev set
inter_dev = np.sum(np.logical_and(y_dev, yhat_dev), axis = (1, 2))+K.epsilon()
union_dev = np.sum(np.logical_or(y_dev, yhat_dev), axis = (1, 2))+K.epsilon()
iou_dev = inter_dev/union_dev
print('\n\r IoU score mean of testing set:\n\r')
print(iou_dev.mean())

# Evaluating IoU score for test set
inter_te = np.sum(np.logical_and(y_te, yhat_te), axis = (1, 2))+K.epsilon()
union_te = np.sum(np.logical_or(y_te, yhat_te), axis = (1, 2))+K.epsilon()
iou_te = inter_te/union_te
print('\n\r IoU score mean of testing set:\n\r')
print(iou_te.mean())
