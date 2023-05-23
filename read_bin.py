import numpy as np

# specify the path to the .bin file
filename = '../output/output3.bin'

# read the binary data as a numpy array of float32 values
data = np.fromfile(filename, dtype=np.float32)
print(data.shape)


# reshape the array into a Nx4 matrix, where N is the number of points
data = data.reshape(-1, 4)

# 加上这句话可以使得终端输出完整的数据
# np.set_printoptions(threshold=np.inf)

# extract the XYZ coordinates from the matrix
xyz = data[:, :3]

print(xyz.shape)
print(f'data is: {xyz}')