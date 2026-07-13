import numpy as np
import h5py
import os
import WHAM_workers as workers
import matplotlib.pyplot as plt

z = np.linspace(0, 100, 1000)
y = np.sin(z)
x = np.cos(z)

path = os.getcwd()
os.mkdir("output")

with h5py.File(os.path.join(path, "output", "data.h5"), 'w') as ff:
    gp = ff.create_group("Test")
    gp.create_dataset('x', data=x)
    gp.create_dataset('y', data=y)
    gp.create_dataset('z', data=z)
    ff.close()
    
with h5py.File(os.path.join(path, "output", "data.h5"), 'r') as ff:
    x_data = ff["Test"]['x'][:]
    y_data = ff["Test"]['y'][:]
    z_data = ff["Test"]['z'][:]
    ff.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data, c=np.arange(len(x_data)), cmap='viridis', s=2, )
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.title("Testing Firebird")
plt.savefig(os.path.join(path, "output", "image.png"))
plt.show()
plt.close()