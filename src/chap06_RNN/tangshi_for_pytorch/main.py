import numpy as np # 导入NumPy库并简称为np
import collections
import torch
import os
from torch.autograd import Variable
import torch.optim as optim

import rnn

start_token = 'B' #定义了起始标记，表示序列的开始
end_token = 'E'#定义了结束标记表示序列的结束
batch_size = 64 #定义了训练或推理时的批处理大小（Batch Size），即每次处理的样本数量


def process_poems1(file_name):
    """
    处理古诗文本文件，返回诗歌的向量表示（每个字映射为索引）。

    :param file_name: 包含诗歌的文件路径，格式为 每行 "标题:内容"
    :return:
        poems_vector: 二维列表，每首诗转换为字的索引序列
        word_int_map: 字到索引的映射字典
        words: 所有字组成的元组，按频率降序排序，最后加一个空格符
    例子：[[1,2,3,4],[5,6,7,8]]
    """

    poems = []  # 存储处理后的诗歌
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                # 尝试按“标题:内容”格式解析
                title, content = line.strip().split(':')
                # 去除空格
                content = content.replace(' ', '')

                # 跳过包含特殊字符或起始/结束标记的诗句
                #if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \ start_token in content or end_token in content
                if any(token in content for token in ['_', '(', '（', '《', '[', start_token, end_token]):
                    continue
                # 跳过长度不合理的诗句
                if len(content) < 5 or len(content) > 80:
                    continue

                # 添加起始和结束标记
                content = start_token + content + end_token
                # 将处理后的诗歌内容添加到列表中
                poems.append(content)
            except ValueError:
                print("error")  # 如果行不符合“标题:内容”格式则跳过
                pass

    # 按诗的长度进行排序，便于后续按批处理
    poems = sorted(poems, key=lambda line: len(line))

    # 统计所有诗句中的字频
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]  # 拆成单字列表

    counter = collections.Counter(all_words)  # 统计每个字的出现次数
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 按频率降序排序

    # 提取所有字，按频率排列，加一个空格符用于补齐
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)

    # 构建字到索引的映射
    word_int_map = dict(zip(words, range(len(words))))

    # 将诗句转为索引序列
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]

    return poems_vector, word_int_map, words


def process_poems2(file_name):
    """
    处理诗歌文本数据，转换为向量表示
    :param file_name: 输入的文本文件名，每行为一首诗
    :return: 
        poems_vector：二维列表，第一维是诗的数量，第二维是每首诗中每个字对应的索引
        word_int_map：字到索引的映射字典
        words：包含所有字的元组，按出现频率排序

    示例：
        poems_vector = [[1, 2, 3, 4], [5, 2, 8, 7]]
    """

    poems = []  # 存储所有符合条件的诗
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()  # 去除首尾空白符
                if line:
                    # 移除空格和常见标点符号
                    content = line.replace(' ', '').replace('，', '').replace('。', '')

                    # 过滤包含特殊字符的诗句
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                        continue

                    # 过滤长度不符合要求的诗句
                    if len(content) < 5 or len(content) > 80:
                        continue

                    # 添加起始符和结束符
                    content = start_token + content + end_token
                    poems.append(content)

            except ValueError:
                # 忽略读取或处理异常
                pass

    # 按诗的长度进行排序（便于后续批处理时填充对齐）
    poems = sorted(poems, key=lambda line: len(line))

    # 统计所有诗中每个字出现的频率
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]

    # 使用Counter统计词频，并按频率降序排序
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    # 提取所有字，添加空格字符用于填充
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)

    # 建立字到索引的映射
    word_int_map = dict(zip(words, range(len(words))))

    # 将所有诗转为索引表示
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]

    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成训练所需的批次数据（x_batches 和 y_batches），并进行对齐填充
    """
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    
    # 填充字符的索引
    pad_idx = word_to_int.get(' ', len(word_to_int))

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batch_data = poems_vec[start_index:end_index]
        
        # 获取当前批次的最大长度
        max_length = max(len(row) for row in batch_data)
        
        x_data = []
        y_data = []
        
        for row in batch_data:
            # 填充到最大长度
            padding = [pad_idx] * (max_length - len(row))
            x_row = row + padding
            
            # y 是 x 的偏移，且最后一个字符通常预测结束符或继续填充
            y_row = row[1:] + [row[-1]] + padding
            
            x_data.append(x_row)
            y_data.append(y_row)

        x_batches.append(x_data)
        y_batches.append(y_data)

    return x_batches, y_batches


def run_training():
    print("开始处理数据集...")
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    print("数据集处理完成, 词汇表大小:", len(word_to_int))
    print("诗歌数量:", len(poems_vector))
    
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(5)
    
    # 初始化模型
    # vocab_len 设为 len(word_to_int) + 1 以包含填充符
    model = rnn.RNN_model(
        vocab_len=len(word_to_int) + 1, 
        embedding_dim=128, 
        lstm_hidden_dim=256,
        num_layers=2
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fun = torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        
        epoch_loss = 0
        for batch in range(n_chunk):
            # 将批次数据转换为 Tensor
            batch_x = torch.LongTensor(batches_inputs[batch]).to(DEVICE)
            batch_y = torch.LongTensor(batches_outputs[batch]).to(DEVICE)

            # 前向传播
            logits, _ = model(batch_x)
            
            # 计算损失：CrossEntropyLoss 期望输入为 (N, C, ...)
            # 展平以便计算
            loss = loss_fun(logits.view(-1, model.vocab_length), batch_y.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # 梯度裁剪
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch}/{n_chunk} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} Average Loss: {epoch_loss / n_chunk:.4f}")
        
        # 每轮保存一次模型
        torch.save(model.state_dict(), './poem_generator_rnn.pth')


def gen_poem(begin_word, word_int_map, vocabularies, model, device, temperature=1.0):
    """
    基于 LSTM 模型生成古诗
    """
    # 将字转换为索引
    word_idx = word_int_map.get(begin_word)
    if word_idx is None:
        word_idx = np.random.randint(0, len(word_int_map))

    poem = [begin_word]
    input_idx = torch.LongTensor([[word_idx]]).to(device)
    hidden = None

    with torch.no_grad():
        for i in range(100): # 最多生成 100 个字
            logits, hidden = model(input_idx, hidden)
            
            # 使用温度采样
            probs = torch.softmax(logits[0, -1] / temperature, dim=0).cpu().numpy()
            word_idx = np.random.choice(len(probs), p=probs)
            
            word = vocabularies[word_idx]
            if word == end_token:
                break
            
            poem.append(word)
            input_idx = torch.LongTensor([[word_idx]]).to(device)

    return "".join(poem)


def pretty_print_poem(poem):
    """
    格式化打印诗歌，每句诗后换行
    """
    # 简单的按标点符号换行
    punctuations = ['，', '。', '！', '？']
    for char in poem:
        print(char, end='')
        if char in punctuations:
            print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='唐诗生成器')
    parser.add_argument('--mode', type=str, default='gen', choices=['train', 'gen'], help='运行模式: train (训练) 或 gen (生成)')
    parser.add_argument('--start', type=str, default='日', help='生成的起始字')
    parser.add_argument('--temp', type=float, default=1.0, help='采样温度')
    args = parser.parse_args()

    if args.mode == 'train':
        run_training()
    else:
        # 优化：只加载一次数据和模型
        print("正在加载数据和模型...")
        poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = rnn.RNN_model(
            vocab_len=len(word_int_map) + 1, 
            embedding_dim=128, 
            lstm_hidden_dim=256,
            num_layers=2
        ).to(DEVICE)

        if os.path.exists('./poem_generator_rnn.pth'):
            model.load_state_dict(torch.load('./poem_generator_rnn.pth', map_location=DEVICE))
            print("[OK] 已加载预训练模型。")
        else:
            print("[Warning] 未找到预训练模型，将使用随机初始化权重（生成内容可能无意义）。")
        
        model.eval()

        # 预设几个起始字进行演示
        seeds = [args.start] if args.start else ['日', '红', '山', '夜', '湖', '君']
        print("="*30)
        print("正在生成古诗示例...")
        print("="*30)
        for seed in seeds:
            poem = gen_poem(seed, word_int_map, vocabularies, model, DEVICE, temperature=args.temp)
            print(f"起始字: {seed}")
            pretty_print_poem(poem)
            print("-" * 20)


