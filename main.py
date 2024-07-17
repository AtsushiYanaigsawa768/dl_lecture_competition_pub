import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import os
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import csv

from torch.optim.lr_scheduler import StepLR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import BertTokenizer, BertModel


def process_text(text):
    """
    Process the given text by performing various transformations.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text after applying the transformations.
    """
    
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
def process_question(question):
    """
    Process the given question by performing various transformations.

    Args:
        question (str): The input question to be processed.

    Returns:
        str: The processed question after applying the transformations.
    """
    # lowercase
    question = question.lower()
    # remove articles
    question = re.sub(r'\b(a|an|the)\b', '', question)
    # remove punctuation
    question = re.sub(r"[^\w\s':]", ' ', question)
    # remove extra spaces
    question = re.sub(r'\s+', ' ', question).strip()
    return question

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, class_mapping=None):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.answer = answer
        self.class_mapping = class_mapping
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        times = 0
        length = 0
        self.max_length = 0
        for question in self.df["question"]:
            question = process_question(question)
            words = question.split(" ")
            times += 1
            for word in words:
                length += 1
                word = process_question(word)
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)
            if self.max_length < length:
                self.max_length = length
            length = 0
        print(self.max_length)
        self.idx2question = {v: k for k, v in self.question2idx.items()}
        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            max_id = max(self.answer2idx.values())
            for answer, class_id in class_mapping.items():
                if answer not in self.answer2idx:
                    max_id += 1
                    self.answer2idx[answer] = max_id
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            # self.answer2idxの最大IDを取得
            max_id = max(self.answer2idx.values())
            # class_mappingのエントリを追加
            for answer, class_id in class_mapping.items():
                if answer not in self.answer2idx:
                    max_id += 1
                    self.answer2idx[answer] = max_id
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        question_words = self.df["question"][idx]
        # question_words = process_question(question_words)
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # question_ids =tokenizer(question_words, return_tensors="pt", max_length=60, padding=True)
        # print(question_ids)
        # print(torch.Tensor(question_ids))
        # tokenizer improve
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        question_tokens = tokenizer.tokenize(question_words,return_tensors="pt", max_length=56, truncation=True, padding="max_length")
        question = tokenizer.convert_tokens_to_ids(question_tokens)
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # print(torch.Tensor(question).numel())
        # question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        # question_words = self.df["question"][idx].split(" ")
        # for word in question_words:
        #     try:
        #         question[self.question2idx[word]] = 1  # one-hot表現に変換
        #     except KeyError:
        #         question[-1] = 1  # 未知語
        # print(torch.Tensor(question).numel())
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            # answer confidence がyesのもののみを使用
            answers_tk = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx] if answer["answer_confidence"] == "yes"]
            mode_answer_idx = mode(answers_tk)  # 最頻値を取得（正解ラベル）
            return image,torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image,  question

    def __len__(self):
        return len(self.df)

    def get_class_mapping(self):
        return self.class_mapping



# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

def ResNet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet152()
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        question_feature = self.text_encoder(question)  # テキストの特徴量

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    step = 0
    for image, question, answers, mode_answer in dataloader:
        # question = question.view(-1, 512)
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)
        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
        # 監視用
        print(f"step: {step}, loss: {loss.item()}")
        step += 1
        

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


def eval(model, dataloader, optimizer, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start
def main(url):
    # deviceの設定
    set_seed(42)
    device = "cpu"
    # zca = ZCAWhitening()
    # # 画像data/trainに存在する画像ごとに、ZCAWhiteningを適用する
    # #  画像の読み込み
    # image_dir = "./data/train"
    # image_files = os.listdir(image_dir)
    # for image_file in image_files:
    #     image_path = os.path.join(image_dir, image_file)
    #     image = Image.open(image_path)
    #     image = image.astype(np.float32)
    #     zca.fit(image)
    #     # ここでZCAWhiteningを適用した画像を保存するなどの処理を行う

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomCrop(32, padding=(4, 4, 4, 4), padding_mode='constant'),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    # CSVファイルからclass_mappingを読み込みます
    class_mapping = pd.read_csv('class_mapping.csv')
    # BERTのトークナイザーとモデルをロードします
    # class_mappingを辞書に変換します
    class_mapping = dict(zip(class_mapping['answer'], class_mapping['class_id']))
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform,class_mapping=class_mapping)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(vocab_size=56, n_answer=len(train_dataset.answer2idx)).to(device)
    print("ATTENTION: model is loaded successfully!")
    # optimizer / criterion
    num_epoch = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    train_times = []
    train_losses = []
    train_accs = []
    train_simple_accs = []

    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15,eta_min=0.00001)
    # print("ATTENTION: start training!")
    for epoch in range(num_epoch):
        # learning rate の調整
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
            f"train simple acc: {train_simple_acc:.4f}")
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"{url}_{epoch+1}_model.pth")
        # 各エポックの値をリストに追加
        train_times.append(train_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_simple_accs.append(train_simple_acc)
    # Train Timeのグラフを保存
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epoch), train_times)
    plt.title('Train Time')
    plt.savefig(f'image/{url}_train_time.png')
    
    # Train Lossのグラフを保存
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epoch), train_losses)
    plt.title('Train Loss')
    plt.savefig(f'image/{url}_train_loss.png')

    # Train Accuracyのグラフを保存
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epoch), train_accs)
    plt.title('Train Accuracy')
    plt.savefig(f'image/{url}_train_accuracy.png')

    # Train Simple Accuracyのグラフを保存
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_epoch), train_simple_accs)
    plt.title('Train Simple Accuracy')
    plt.savefig(f'image/{url}_train_simple_accuracy.png')
    # データを準備
    data = [train_times, train_losses, train_accs, train_simple_accs]

    # CSVファイルに書き込む
    with open(f'{url}_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(*data))
    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), f"{url}_model.pth")
    np.save(f"{url}_submission.npy", submission)
    # modelをnp.load の後にsubmission.npyを作成する


if __name__ == "__main__":
    main("bert")
