'''final version of the torusReader by Ahmet Efe, Sven Leschber, Mika Rother'''

from tensorflow import keras
import tensorflow as tf
from keras.models import model_from_json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import numpy as np
import json
from keras.models import model_from_json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

# load dataset from csv files
xy_test = np.loadtxt('xy_test.csv', delimiter=',')
z_test = np.loadtxt('z_test.csv', delimiter=',')

# load torus model from JSON file and create Torus model with best weights
json_file = open('bestTorus.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
bestTorus = tf.keras.models.model_from_json(loaded_model_json)
bestTorus.load_weights("weightsTorus.best.hdf5")
print('-----------------------------------------------------------')
print("Loaded model from disk")
print('-----------------------------------------------------------')

# evaluate loaded model on test data
bestTorus.compile(optimizer = tf.optimizers.Adamax(), loss='logcosh')
results = bestTorus.evaluate(xy_test, z_test, verbose=0)
print('Torus: test loss, test acc:', results)
print('-----------------------------------------------------------')

theta = np.linspace(0,np.pi, 100)
phi = np.linspace(0,2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)
zp = deepcopy(z)
zm = deepcopy(z)

buffer = []
for a in range(100):
    buffer1 = []
    for b in range(100):
        bestTorus.predict([[x[a][b],y[a][b]]])
print('-----------------------------------------------------------')

for i in range(len(z)):
    for j in range(len(buffer)):
        zp[i-1][j-1] = buffer[i-1][j-1]
        zm[i-1][j-1] = buffer[i-1][j-1]*-1

# plot the torus to see how good the approximation is
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_zlim(-3,3)
ax1.plot_wireframe(x, y, zp, rstride=5, cstride=5, color='k')
ax1.plot_wireframe(x, y, zm, rstride=5, cstride=5, color='k')
ax1.view_init(36, 26)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_zlim(-3,3)
ax2.plot_wireframe(x, y, zp, rstride=5, cstride=5, color='k')
ax2.plot_wireframe(x, y, zm, rstride=5, cstride=5, color='k')
ax2.view_init(0, 0)
ax2.set_xticks([])
plt.show()