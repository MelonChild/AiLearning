import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # homework
        # 计算对每一个 context而言 哪些query和它最相关
        # 相似度矩阵 softmax 对列归一化
        # 计算quer向量加权和
        c2q_att = tf.keras.activations.softmax(similarity,axis=-1)
        encode_q = tf.keras.backend.expand_dims(qencode,axis=1)
        c2q_att = tf.keras.backend.expand_dims(c2q_att,axis=-1)*encode_q

        c2q_att = tf.keras.backend.sum(c2q_att,-2)

        print("C2QAttention",similarity,qencode)
        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        # homework
        # 计算对每一个 query 而言 哪些  context 和它最相关
        # 相似度矩阵 每列最大值 softmax归一化 加权计算context向量
        max_similarity = tf.keras.backend.max(similarity,axis=-1)
        attention = tf.keras.activations.softmax(max_similarity,axis=-1)
        expand_attention = tf.keras.backend.expand_dims(attention,axis=-1)*cencode
        weighted_sum = tf.keras.backend.sum(expand_attention,-2)
        expand_weighted_sum = tf.keras.backend.expand_dims(weighted_sum,1)
        num_repeatations = tf.keras.backend.shape(cencode)[1]
        q2c_att = tf.keras.backend.tile(expand_weighted_sum,[1,num_repeatations,1])
        print("Q2CAttention",similarity,cencode)
        return q2c_att
