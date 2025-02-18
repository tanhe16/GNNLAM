from clr import cyclic_learning_rate
import tensorflow as tf


# Optimizer 类：这个类负责为模型定义优化器。
# 它接受多个参数，包括模型本身（model）、预测值（preds）、标签（labels）、学习率（lr）、num_u=269行--drug数量、num_v=598列--disease数量、以及关联数量（association_nam）。
class Optimizer():
    def __init__(self, model, preds, labels, lr, num_u, num_v, association_nam):
        norm = num_u*num_v / float((num_u*num_v-association_nam) * 2)  # 计算权重正则化因子
        # 获取预测值和标签
        preds_sub = preds
        labels_sub = labels

        # 计算正样本权重
        pos_weight = float(num_u*num_v-association_nam)/(association_nam)
        global_step = tf.Variable(0, trainable=False)  # 定义全局步数变量

        # 使用Adam优化器，设置学习率----------------采用的是循环学习率，clr.py，使学习率在最大学习率和最小学习率之间发生变化，帮助我们平衡训练速度和ACC。
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=global_step, learning_rate=lr*0.1,

                                                                                   max_lr=lr, mode='exp_range', gamma=.99)) #0.995--0.99
        # 计算损失函数---------加权交叉熵----------对应的是公式（11）
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        # 使用优化器最小化损失函数
        self.opt_op = self.optimizer.minimize(
            self.cost, global_step=global_step,)
        # 计算梯度
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
