
import math
import os
import random

import pandas as pd
# import tensorflow as tf
# from tensorflow.python.keras import Input, Model
# from tensorflow.python.keras.applications.densenet import DenseNet169, DenseNet121
# from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.python.keras.applications.xception import Xception
# from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
# from tensorflow.python.keras.applications.resnet import ResNet50, ResNet152
# from tensorflow.python.keras.layers import GlobalAveragePooling2D, BatchNormalization
# from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Flatten, GlobalMaxPooling2D, \
#     MaxPooling2D, Conv2D
from tensorflow.python.keras import regularizers, Input, Model
from tensorflow.keras.applications import ResNet152, Xception
from tensorflow.python.keras.api.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.layers import InputSpec, Concatenate, Dense, Dropout, GlobalAveragePooling2D, Reshape, Conv2D, \
    Flatten, GlobalMaxPooling2D
from tensorflow.keras import backend as K, Input, Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.layers import BatchNormalization
# from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Rescaling
from util import read_data, get_data, get_data_new, get_data_newfour
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Concatenate, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import keras_metrics as km
from util import read_data, get_data, get_data_new
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()##.model
        #val_predict1 = (np.asarray(self.model.predict(self.validation_data[1][0]))).round()##.model
        #val_predict2 = (np.asarray(self.model.predict(self.validation_data[2][0]))).round()##.model
        #val_predict3 = (np.asarray(self.model.predict(self.validation_data[3][0]))).round()##.model
        #val_predict4 = (np.asarray(self.model.predict(self.validation_data[4][0]))).round()##.model
        #val_predict5 = (np.asarray(self.model.predict(self.validation_data[5][0]))).round()##.model
        #val_predict6 = (np.asarray(self.model.predict(self.validation_data[6][0]))).round()##.model

        val_targ = self.validation_data[1]###.model

        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average=None)###
        _val_precision = precision_score(val_targ, val_predict,average=None)###
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f "%_val_f1)


def CoordAtt(x, reduction=32):
    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    print(x_shape)
    [b, h, w, c] = x_shape
    x_h = AvgPool2D(pool_size=(1, w), strides=1)(x)
    x_w = AvgPool2D(pool_size=(h, 1), strides=1)(x)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(mip, (1, 1), strides=1, activation=coord_act, name='ca_conv1')(y)

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = Conv2D(c, (1, 1), strides=1, activation=tf.nn.sigmoid, name='ca_conv2')(x_h)
    a_w = Conv2D(c, (1, 1), strides=1, activation=tf.nn.sigmoid, name='ca_conv3')(x_w)

    out = x * a_h * a_w

    return out

class MixedPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), padding='same', data_format='channels_last', **kwargs):
        super(MixedPooling2D, self).__init__(**kwargs)

        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(pool_size, 2, 'strides')
        self.padding = padding
        self.data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        self.input_spec = InputSpec(ndim=4)
        self.alpha = random.uniform(0, 1)
        self.alpha_frequencies = np.zeros(2)

    def build(self, input_shape):
        super(MixedPooling2D, self).build(input_shape)

    def _pooling_function(self, x, name=None):
        """
        Mixed Pooling
        """
        max_pool = K.pool2d(x, self.pool_size, strides=self.strides, padding=self.padding, pool_mode="max")
        avg_pool = K.pool2d(x, self.pool_size, strides=self.strides, padding=self.padding, pool_mode="avg")

        def _train_pool(max_pool, avg_pool):
            self.alpha = random.uniform(0, 1)
            self.alpha_frequencies[0] += self.alpha
            self.alpha_frequencies[1] += 1 - self.alpha

            return self.alpha * max_pool + (1 - self.alpha) * avg_pool

        def _test_pool(max_pool, avg_pool):
            return K.switch(K.less(self.alpha_frequencies[0], self.alpha_frequencies[1]), avg_pool, max_pool)

        outs = K.in_train_phase(_train_pool(max_pool, avg_pool), _test_pool(max_pool, avg_pool))

        return outs

    def compute_output_shape(self, input_shape):
        r, c = input_shape[1], input_shape[2]
        sr, sc = self.strides
        num_r = math.ceil(r / sr) if self.padding == 'same' else r // sr
        num_c = math.ceil(c / sc) if self.padding == 'same' else c // sc
        return (input_shape[0], num_r, num_c, input_shape[3])

    def call(self, inputs):
        output = self._pooling_function(inputs)
        return output

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'strides': self.strides
        }
        base_config = super(MixedPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def exponent(global_epoch,
            learning_rate_base,
            decay_rate,
            min_learn_rate=0,
            ):

    learning_rate = learning_rate_base * pow(decay_rate, global_epoch)
    learning_rate = max(learning_rate,min_learn_rate)
    return learning_rate

class ExponentDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 decay_rate,
                 global_epoch_init=0,
                 min_learn_rate=0,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 全局初始化epoch
        self.global_epoch = global_epoch_init

        self.decay_rate = decay_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    def on_epoch_end(self, epochs ,logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
	#更新学习率
    def on_epoch_begin(self, batch, logs=None):
        lr = exponent(global_epoch=self.global_epoch,
                    learning_rate_base=self.learning_rate_base,
                    decay_rate = self.decay_rate,
                    min_learn_rate = self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_epoch + 1, lr))

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    train_ds, val_ds = get_data_newfour()

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
    normalization_layer = Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    input_layer = Input(shape=(300, 300, 3))
    resNet = ResNet152(include_top=False, weights=None, input_tensor=input_layer,
                       input_shape=(300, 300, 3))
    xception = Xception(include_top=False, weights=None, input_tensor=input_layer,
                        input_shape=(300, 300, 3))
    top1_model = MixedPooling2D(data_format='channels_last')(resNet.output)
    top2_model = MixedPooling2D(data_format='channels_last')(xception.output)
    concatenate_model = Concatenate(axis=1)([top1_model, top2_model])

    concatenate_model.trainable = False
    out = CoordAtt(concatenate_model)
    out = Flatten()(out)
    top_model = Dense(units=512, activation="relu", kernel_regularizer='l2')(out)
    top_model = Dense(units=256, activation="relu", kernel_regularizer='l2')(top_model)
    top_model = Dropout(rate=0.5)(top_model)
    top_model = Dense(units=2, activation="softmax")(top_model)
    model = Model(inputs=input_layer, outputs=top_model)

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy',km.f1_score(),km.recall(),km.precision()])

    early_stopping = EarlyStopping(

        monitor = 'val_accuracy',
        verbose = 1,
        patience = 20,
        restore_best_weights = True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(min_lr=0.00001,
                                                     factor=0.2)
    # 设置训练参数
    # epochs = 10
    init_epoch = 0
    # 每一次训练使用多少个Batch
    batch_size = 64
    # 最大学习率
    learning_rate_base = 1e-3
    sample_count = 1779
    # 学习率
    exponent_lr = ExponentDecayScheduler(learning_rate_base = learning_rate_base,
                                        global_epoch_init = init_epoch,
                                        decay_rate = 0.9,
                                        min_learn_rate = 1e-6
                                        )
    num_0 = len(os.listdir('train_data_all/0'))
    num_1 = len(os.listdir('train_data_all/1'))

    total = num_0 + num_1
    weight_for_0 = total / num_0 / 2.0
    weight_for_1 = total / num_1 / 2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)

    history = model.fit(train_ds, epochs=1000, callbacks=[early_stopping, exponent_lr],validation_data=val_ds, class_weight=class_weight)
    predict = []
    val_targ =[]
    for i in range(33):
       res = model.predict(val_ds[i][0])
       [predict.append(np.argmax(r)) for r in res]
       [val_targ.append(np.argmax(label)) for label in val_ds[i][1]]

    print(len(predict))
    print(len(val_targ))
    print(predict)
    print(val_targ)
    
    _val_f1 = f1_score(val_targ, predict,average='micro')
    _val_recall = recall_score(val_targ, predict,average=None)###
    _val_precision = precision_score(val_targ, predict,average=None)###
    print('_val_f1',_val_f1)
    print('_val_recall',_val_recall[0])
    print('_val_precision',_val_precision[0])
  
    hist_df = pd.DataFrame(history.history) 
      
    y_pre_file = 'end_model_predict.csv'  
    y_rel_file = 'end_model_reltag.csv' 
    test1=pd.DataFrame(data=predict)
    test2=pd.DataFrame(data=val_targ)
    test1.to_csv(y_pre_file, encoding= 'utf-8')
    test2.to_csv(y_rel_file, encoding= 'utf-8')

    
    hist_csv_file = 'end_model_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # model.save('end_model.h5')
