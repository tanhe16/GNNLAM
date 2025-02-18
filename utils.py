import numpy as np
import tensorflow as tf
import scipy.sparse as sp


# 这个函数用于初始化权重，采用了Xavier/Glorot初始化方法。
# 它接受输入维度和输出维度，并返回一个根据 Glorot 初始化方法初始化的权重变量。
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


# 这个函数实现了对稀疏张量进行dropout操作。
# 它接受稀疏输入张量 x、保留的元素比例 keep_prob 和张量中非零元素的数量 num_nonzero_elems。
# 在这个函数中，根据 keep_prob 计算一个随机张量，并将其转换为布尔类型的dropout掩码，然后使用 tf.sparse_retain 函数应用dropout操作。
def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)


# 这个函数将稀疏矩阵表示转换为元组表示。
# 如果输入不是COO格式的稀疏矩阵，则将其转换为COO格式。
# 最后，返回元组 (coords, values, shape)，其中 coords 是非零元素的坐标，values 是非零元素的值，shape 是矩阵的形状。
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# 这个函数对输入的邻接矩阵进行预处理。
# 首先将邻接矩阵转换为COO格式，然后计算每行的和，接着根据度矩阵的倒数的平方来归一化邻接矩阵，最后将归一化后的邻接矩阵转换为元组表示。
def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


# 这个函数根据给定的药物-疾病关联矩阵构建一个完整的图结构的邻接矩阵。
# 首先创建两个全零矩阵，分别表示药物和疾病的邻接矩阵。
# 然后将给定的药物-疾病关联矩阵水平堆叠起来，得到一个 mat1。
# 将 mat1 转置后与全零矩阵垂直堆叠，得到完整的邻接矩阵 adj
def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


# 这个函数与 constructNet 函数类似，不过它接受额外的药物和疾病的邻接矩阵作为参数。
# 它将给定的药物-疾病关联矩阵与额外的药物和疾病的邻接矩阵进行水平和垂直堆叠，构建一个更复杂的图结构的邻接矩阵。
def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


# 这些函数主要用于构建和处理图数据，其中 constructNet 和 constructHNet 函数直接体现了图结构的构建过程，
# 而其他函数则为构建过程提供了必要的辅助功能，如权重初始化、dropout操作和稀疏矩阵的处理。