from tensorflow.keras.callbacks import ModelCheckpoint

from DataProcess import load_data
from Model.densenet import get_compiled
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 变量初始化
NUM_CLASSES = 2
batch_size = 16
epoch = 30

x_train, y_train = load_data(True)
x_val, y_val = load_data()
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

tf.compat.v1.disable_eager_execution()

# compile model
model = get_compiled()
# model_path = 'tmp/weights0618.09.h5'
# model.load_weights(model_path)

# train
checkpointer = ModelCheckpoint(filepath='tmp/weights0621.{epoch:02d}.h5', verbose=1,
                                period=1, save_weights_only=True)
model.fit(x_train,
          y_train,
          epochs=epoch,
          validation_data=(x_val, y_val),
          shuffle=True,
          batch_size=batch_size,
          callbacks=[checkpointer]
          )
