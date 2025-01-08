import numpy as np
import matplotlib.pyplot as plt


train_data1 = np.load('data0.npy')
train_lab1 = np.load('lab0.npy')

i = 42
plt.imshow(train_data1[i])
plt.savefig('img.png')
print(train_lab1[i])