# 引入必要的数据库
import torch.nn as nn # 导入PyTorch的神经网络模块(nn) 并将它简称为"nn"
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# 对神经网络中的线性层（Linear）进行权重初始化
def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())# 获取权重张量的形状: [输出维度, 输入维度]
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # 计算权重初始化范围
        m.weight.data.uniform_(-w_bound, w_bound) # 均匀分布初始化权重
        m.bias.data.fill_(0) # 偏置置零
        print("inital  linear weight ")# 打印初始化信息


# 定义词嵌入模块
class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)

    def forward(self, input_sentence):
        """
        :param input_sentence: 词索引张量 [batch_size, seq_len] 或 [seq_len]
        :return: 嵌入向量张量 [batch_size, seq_len, embedding_dim]
        """
        return self.word_embedding(input_sentence)


# 定义基于 LSTM 的 RNN 模型
class RNN_model(nn.Module):
    def __init__(self, vocab_len, embedding_dim, lstm_hidden_dim, num_layers=2):
        super(RNN_model, self).__init__()

        # 模型参数
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.num_layers = num_layers

        # 1. 词嵌入层：将词索引转换为高维稠密向量
        self.embeddings = nn.Embedding(vocab_len, embedding_dim)

        # 2. LSTM 层：处理序列特征
        # batch_first=True：输入输出维度为 (batch, seq, feature)
        # dropout=0.2：在层间引入正则化，防止过拟合
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 3. 输出层：将 LSTM 隐藏状态映射回词汇表大小
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)

        # 初始化权重
        self.apply(weights_init)

    def forward(self, x, hidden=None):
        """
        前向传播
        Args:
            x: 输入序列索引 [batch_size, seq_len]
            hidden: 初始隐藏状态 (h, c)，若为 None 则自动初始化为 0
        Returns:
            output: 每个位置的预测分布 [batch_size, seq_len, vocab_len]
            hidden: 更新后的隐藏状态
        """
        batch_size = x.size(0)

        # 词嵌入查找
        embeds = self.embeddings(x)  # [batch, seq, embed_dim]

        # LSTM 运算
        # output 形状: [batch, seq, hidden_dim]
        output, hidden = self.lstm(embeds, hidden)

        # 展平以便输入全连接层
        # out 形状: [batch * seq, hidden_dim]
        out = output.contiguous().view(-1, self.lstm_dim)

        # 映射到词汇表空间
        # logits 形状: [batch * seq, vocab_len]
        logits = self.fc(out)

        # 恢复维度：[batch, seq, vocab_len]
        logits = logits.view(batch_size, -1, self.vocab_length)

        return logits, hidden

