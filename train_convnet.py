# coding: utf-8
import sys, os

from common.util import smooth_curve

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

from common.optimizer import *

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 30

# 최적화기 초기화
optimizers = {
    'SGD': SGD(),
    'Momentum': Momentum(),
    'AdaGrad': AdaGrad(),
    'Adam': Adam()
}

markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_epochs)

# 각 최적화기별로 손실 값을 저장할 딕셔너리 생성
losses = {key: [] for key in optimizers.keys()}

# 각 최적화기로 네트워크를 훈련
for key in optimizers.keys():
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer=key, optimizer_param={'lr': 0.005},
                      evaluate_sample_num_per_epoch=1000)

    for i in range(trainer.max_iter):
        loss = trainer.train_step()
        losses[key].append(loss)

# 손실 값을 그래프로 표시
for key in optimizers.keys():
    plt.plot(np.arange(len(losses[key])), smooth_curve(np.array(losses[key])), marker=markers[key], markevery=100,
             label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()