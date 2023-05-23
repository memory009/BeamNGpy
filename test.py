import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss(y, p):
    return -(y*np.log(p) + (1-y)*np.log(1-p))

y = 1  # y=1, 红色曲线
p = np.linspace(0, 1, 100)
loss = cross_entropy_loss(y, p)
plt.plot(p, loss, 'r')

y = 0  # y=0, 蓝色曲线
p = np.linspace(0, 1, 100)
loss = cross_entropy_loss(y, p)
plt.plot(p, loss, 'b')

plt.xlabel('Prediction probability')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss')
plt.legend(['y=1', 'y=0'])
plt.show()
