from model13 import *
from data import *
os.environ["CUDA_VISIBLE_DEVICES"]="3"

nbatch = 16

x_tr = np.load(data_path+'x_tr.npy')
y_tr = np.load(data_path+'y_tr.npy')
x_dev = np.load(data_path+'x_dev.npy')
y_dev = np.load(data_path+'y_dev.npy')
print(x_tr.shape)
print(x_dev.shape)
model = unet()
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)

#history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
#model.save('model_10ep.h5')
#training_loss = history.history['loss']
#test_loss = history.history['val_loss']
#np.savetxt('trainingloss10.txt', training_loss, delimiter = " ", fmt = "%s")
#np.savetxt('testloss10.txt', test_loss, delimiter = " ", fmt = "%s")
"""
history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=100, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
model.save('model_100ep.h5')
training_loss = history.history['loss']
test_loss = history.history['val_loss']
np.savetxt('trainingloss100.txt', training_loss, delimiter = " ", fmt = "%s")
np.savetxt('testloss100.txt', test_loss, delimiter = " ", fmt = "%s")


model.fit(x_tr, y_tr, batch_size=8, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_te, y_te))
model.save('model_10ep.h5')
"""
for i in range(10):
    if i != 0:
        model = load_model('model_'+str(10*(i))+'ep.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    history = model.fit(x_tr, y_tr, batch_size=nbatch, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint], validation_data=(x_dev, y_dev))
    model.save('model_'+str(10*(i+1))+'ep.h5')
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    np.savetxt('trainingloss' + str(10*(i+1))+'.txt', training_loss, delimiter = " ", fmt = "%s")
    np.savetxt('testloss' + str(10*(i+1))+'.txt', test_loss, delimiter = " ", fmt = "%s")

