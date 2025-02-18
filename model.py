import tensorflow as tf
from layers import GraphConvolutionSparse, InnerProductDecoder, GraphAttention, GraphSAGE
from utils import *

# GCNModel 类：这个类是整个模型的主体，
# 初始化时接收了一系列占位符 placeholders，包括特征矩阵、邻接矩阵、dropout率等。
# 它还接收了一些参数，如特征维度num_features、嵌入维度emb_dim、非零特征数量features_nonzero、非零邻接矩阵元素数量adj_nonzero、关系数量num_r 和激活函数 act。
# 在构建过程中，它调用了 build() 方法来建立模型。
class GCNModel():

    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, name, act=tf.nn.elu):
        self.name = name
        self.inputs = placeholders['features']  # 特征占位符（用于传递特征数据）
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']  # 邻接矩阵占位符
        self.dropout = placeholders['dropout']  # （规则丢失率0.4）占位符（控制模型的邻接矩阵的节点丢失率，默认值为 0.0（即不丢失））
        self.adjdp = placeholders['adjdp']  # 邻接矩阵（节点丢失率0.6）占位符
        self.act = act  # 传入激活函数
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))  # 调节不同层输出在嵌入表示中的权重（引入注意机制进行加权）
        self.num_r = num_r  # (269*598)
        with tf.variable_scope(self.name):
            self.build()

    # build() 方法：这个方法定义了模型的具体结构。
    # 首先，对邻接矩阵进行dropout处理。然后，通过 GraphConvolutionSparse 类构建了第一层稀疏图卷积层，接着构建了两个密集图卷积层，
    # 最后将三个图卷积层的输出加权相加得到嵌入表示。最后，通过 InnerProductDecoder 类构建了一个内积解码器层，用于生成重构图。
    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)

        self.hidden1 = GraphConvolutionSparse(               # 稀疏图卷积层
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)

        self.hidden2 = GraphSAGE(
            name='graphsage_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act,
            aggregator_type='mean')(self.hidden1)

        self.emb = GraphAttention(
            name='gat_layer1',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,
            act=self.act)(self.hidden2)


        # 将三个图卷积层的输出（self.hidden1、self.hidden2和self.emb）加权相加得到嵌入表示（self.embeddings）
        self.embeddings = self.hidden1 * \
            self.att[0]+self.hidden2*self.att[1]+self.emb*self.att[2]  # self.att[0]、self.att[1] 和 self.att[2] 是三个权重参数

        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.embeddings)
        # self.embeddings: 模型的嵌入表示，通过加权相加三个图卷积层的输出得到。
        # self.reconstructions: 重构的图，通过内积解码器层得到，用于链接预测任务。


##############################
# 一共三步：3层卷积编码器+注意机制权重+解码器