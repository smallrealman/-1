import torch.nn as nn
import torch
from keras.src.utils import pad_sequences


class SentimentNet(nn.Module):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, vocab_size, input_dim, hid_dim, layers, output_dim):
        super(SentimentNet, self).__init__()
        self.n_layers = layers
        self.hidden_dim = hid_dim
        self.embeding_dim = input_dim
        self.output_dim = output_dim
        drop_prob = 0.5

        self.lstm = nn.LSTM(self.embeding_dim, self.hidden_dim, self.n_layers,
                            dropout=drop_prob, batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)

        self.embedding = nn.Embedding(vocab_size, self.embeding_dim)

    def forward(self, x, hidden):
        x = x.long()
        embeds = self.embedding(x)

        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out[:, -1, :]
        out = out.squeeze()
        out = out.contiguous().view(-1)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import jieba
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data_path = 'E:\\pythonProject\\人工智能\\data_corpus.csv'
    df = pd.read_csv(data_path)

    x = df['evaluation']
    y = df['label']

    texts_cut = [jieba.lcut(one_text) for one_text in x]

    label_set = set()
    for label in y:
        label_set.add(label)
    label_set = np.array(list(label_set))

    labels_one_hot = []
    for label in y:
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label)] = 1
        labels_one_hot.append(label_zero)
    labels = np.array(labels_one_hot)


    num_words = 3000
    tokenizer = jieba.Tokenizer(num_words)
    tokenizer.fit_on_texts(texts=texts_cut)
    num_words = min(num_words, len(tokenizer.word_index) + 1)

    from collections import Counter
    all_words = [word for text in texts_cut for word in text]
    word_counts = Counter(all_words)
    most_common_words = [word for word, count in word_counts.most_common(num_words)]
    word_index = {word: index + 1 for index, word in enumerate(most_common_words)}
    word_index['PAD'] = 0
    word_index['UNK'] = len(word_index)
    sequences = []
    for text in texts_cut:
        sequence = [word_index.get(word, word_index['UNK']) for word in text]
        sequences.append(sequence)

    sentence_len = 64
    texts_seq = tokenizer.texts_to_sequences(texts=texts_cut)
    texts_pad_seq = pad_sequences(texts_seq, maxlen=sentence_len, padding='post', truncating='post')

    # 拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(texts_pad_seq, labels, test_size=0.2, random_state=1)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SentimentNet(num_words, 256, 128, 8, 2)

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    epochs = 32
    step = 0
    epoch_loss_list = []
    model.train()  # 开启训练模式

    for epoch in range(epochs):
        epoch_loss = 0
        for index, (x_train, y_train) in enumerate(train_loader):
            cur_batch = len(x_train)
            h = model.init_hidden(cur_batch)  # 初始化第一个Hidden_state

            x_train, y_train = x_train.to(device), y_train.to(device)
            step += 1  # 训练次数+1

            x_input = x_train.to(device)
            model.zero_grad()

            output, h = model(x_input, h)

            # 计算损失
            loss = criterion(output, y_train.float().view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()

        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Step: {}...".format(step),
              "Loss: {:.6f}...".format(epoch_loss))

        epoch_loss_list.append(epoch_loss)

    model.eval()
    loss = 0
    for data in tqdm(test_loader):
        x_val, y_val = data
        x_val, y_val = x_val.to(device), y_val.to(device)

        h = model.init_hidden(len(x_val))  # 初始化第一个Hidden_state

        x_input = x_val.long()
        x_input = x_input.to(device)
        output, h = model(x_input, h)

        loss += criterion(output, y_val.float().view(-1))

    print("test Loss: {:.6f}...".format(loss))

    test_text_cut = [jieba.lcut("商品质量相当不错，点赞"),
                     jieba.lcut("什么破东西，简直没法使用")]

    test_seq = tokenizer.texts_to_sequences(texts=test_text_cut)
    test_pad_seq = pad_sequences(test_seq, maxlen=sentence_len, padding='post', truncating='post')
    h = model.init_hidden(len(test_pad_seq))

    output, h = model(torch.tensor(test_pad_seq), h)
    print(output.view(-1, 2))

    x = [epoch + 1 for epoch in range(epochs)]
    plt.plot(x, epoch_loss_list)

    plt.xlim(0, 32)
    plt.ylim(0, 100)
    plt.show()

