import numpy as np
import pandas as pd
import mxnet as mx
import logging
from sklearn.cross_validation import train_test_split
from lenet import get_lenet

# create the training 
dataset = pd.read_csv("train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:, 1 : ].values.astype('float32')

# Normalize data
train /= 256.0
print(train.shape)

# split dataset
(train_data, val_data, train_label, val_label) = \
	train_test_split(train, target, test_size = 0.1, random_state = 10)
train_data = np.array(train_data).reshape((-1, 1, 28, 28))
val_data = np.array(val_data).reshape((-1, 1, 28, 28))

batch_size = 500
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size = batch_size, shuffle = True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size = batch_size)

# logging
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level = logging.DEBUG, format = head)

# create model 
devs = mx.cpu()
network=get_lenet(10)
print('network success!')
model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = 15,
        learning_rate      = 0.1,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34)
        )

print('network prepared!')
		
eval_metrics = ['accuracy']
model.fit(
	X = train_iter, 
	eval_metric = eval_metrics,
	eval_data = val_iter
	)

#predict
test = pd.read_csv("test.csv").values
test_data = test.astype('float32')
test_data = np.array(test_data).reshape((-1, 1, 28, 28)) / 256.0
test_iter = mx.io.NDArrayIter(test_data, batch_size = batch_size)

pred = model.predict(X = test_iter)
pred = np.argsort(pred)
np.savetxt('yzp_lenet.csv', np.c_[range(1, len(test) + 1),pred[:, 9]], delimiter = ',', header = 'ImageId,Label', comments = '', fmt = '%d')