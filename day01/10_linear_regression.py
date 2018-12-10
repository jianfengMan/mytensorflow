import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

# 立刻下载数据集
housing = fetch_california_housing(data_home=None, download_if_missing=True)

# 获得X数据行数和列数
m, n = housing.data.shape
print(m,n)
# 20640 8


# 这里添加一个额外的bias输入特征(x0=1)到所有的训练数据上面，因为使用的numpy所有会立即执行
# np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
print(housing_data_plus_bias)


# 创建两个TensorFlow常量节点X和y，去持有数据和标签
# reshape(-1,1) 把未知的数据变成(n,1)
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# 使用一些TensorFlow框架提供的矩阵操作去求theta
XT = tf.transpose(X)
# tf.transpose(input, [dimension_1, dimenaion_2,..,dimension_n]):
# 这个函数主要适用于交换输入张量的不同维度用的，如果输入张量是二维，就相当是转置。
# dimension_n是整数，如果张量是三维，就是用0,1,2来表示。这个列表里的每个数对应相应的维度。
# 如果是[2,1,0]，就把输入张量的第三维度和第一维度交换。

# 解析解一步计算出最优解
# 1.tf.matrix_diag(dia)：输入参数是dia，如果输入时一个向量，那就生成二维的对角矩阵，以此类推
#
# 2.tf.matrix_inverse(A)：输入如果是一个矩阵，就是得到逆矩阵，依次类推，只是输入的A中的元素需要是浮点数，比如tf.float32等格式，如果是整形，就会出错哈。


theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()  # sess.run(theta)
    print(theta_value)

# theta = (XXT)-1XTy ,就是解析解