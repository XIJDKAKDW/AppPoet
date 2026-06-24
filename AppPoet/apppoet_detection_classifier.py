#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest
@File    ：apppoet_detection_classifier.py
@IDE     ：PyCharm
@Author  ：常晓松
@Date    ：2026/1/15 10:23
'''
# coding=utf-8
import platform
import sys

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AppPoet")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")

from pr2_new_2.test001 import load_seqs_from_file
from pr2_new_2.test001Method_new_9_4_2 import get_connection

"""
AppPoet检测分类器模块
根据描述训练深度学习模型进行检测
对应论文第4.3节
"""

import json
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import torch
import numpy as np
import gensim.downloader as api
from collections import defaultdict
import re

class LightweightEmbeddingModel:
    """
    轻量级词嵌入模型
    使用预训练的GloVe或Word2Vec模型
    """

    def __init__(self,
                 model_name: str = 'glove-wiki-gigaword-100',
                 embedding_dim: int = 100,
                 max_words: int = 200,
                 device: Optional[str] = None):
        """
        初始化轻量级嵌入模型

        Args:
            model_name: 预训练模型名称
                - 'glove-wiki-gigaword-50' (50维)
                - 'glove-wiki-gigaword-100' (100维) [默认]
                - 'glove-wiki-gigaword-200' (200维)
                - 'glove-wiki-gigaword-300' (300维)
                - 'word2vec-google-news-300' (300维，较大)
                - 'fasttext-wiki-news-subwords-300' (300维)
            embedding_dim: 嵌入维度
            max_words: 最大词数
            device: 设备 (cuda/cpu)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_words = max_words
        self.device = device

        # 词向量缓存
        self.word_vectors = None
        self.unknown_tokens = defaultdict(int)

        # 初始化模型
        self._initialize_model()

        # 特殊标记
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        # 为特殊标记创建向量
        self.special_vectors = {
            self.pad_token: np.zeros(self.embedding_dim),
            self.unk_token: np.random.normal(scale=0.1, size=self.embedding_dim)
        }

    def _initialize_model(self):
        """加载预训练的词向量模型"""
        try:
            print(f"正在加载预训练模型: {self.model_name}")

            # 根据模型名称选择不同的加载方式
            if 'glove' in self.model_name:
                # GloVe模型
                self.word_vectors = api.load(self.model_name)
            elif 'word2vec' in self.model_name or 'fasttext' in self.model_name:
                # Word2Vec或FastText模型
                self.word_vectors = api.load(self.model_name)
            else:
                # 默认使用GloVe
                self.word_vectors = api.load('glove-wiki-gigaword-100')

            # 获取实际的嵌入维度
            sample_vec = self.word_vectors['the']
            actual_dim = len(sample_vec)
            if actual_dim != self.embedding_dim:
                print(f"警告: 模型维度({actual_dim})与指定维度({self.embedding_dim})不匹配，使用模型维度")
                self.embedding_dim = actual_dim

            print(f"成功加载模型: {self.model_name}")
            print(f"词汇表大小: {len(self.word_vectors.key_to_index)}")
            print(f"嵌入维度: {self.embedding_dim}")

        except Exception as e:
            print(f"加载预训练模型失败: {e}")
            print("将使用随机初始化的嵌入")
            self.word_vectors = None

    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理
        """
        if not text:
            return []

        # 转换为小写
        text = text.lower()

        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^\w\s]', ' ', text)

        # 分割为单词
        words = text.split()

        # 限制最大词数
        words = words[:self.max_words]

        return words

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        获取单个词的向量
        """
        if word in self.special_vectors:
            return self.special_vectors[word]

        if self.word_vectors is not None:
            try:
                if word in self.word_vectors:
                    return self.word_vectors[word]
            except:
                pass

        # 未知词处理
        self.unknown_tokens[word] += 1

        # 为未知词生成随机向量
        if word not in self.special_vectors:
            self.special_vectors[word] = np.random.normal(scale=0.1, size=self.embedding_dim)

        return self.special_vectors[word]

    def embed_text(self, text: str, pooling: str = 'mean') -> np.ndarray:
        """
        嵌入单个文本

        Args:
            text: 输入文本
            pooling: 池化策略 ('mean', 'max', 'sum')
        Returns:
            嵌入向量
        """
        # 预处理文本
        words = self.preprocess_text(text)

        if not words:
            # 返回零向量
            return np.zeros(self.embedding_dim)

        # 获取每个词的向量
        word_vectors = []
        for word in words:
            vec = self.get_word_vector(word)
            word_vectors.append(vec)

        # 转换为numpy数组
        word_vectors = np.array(word_vectors)

        # 池化操作
        if pooling == 'mean':
            embedding = np.mean(word_vectors, axis=0)
        elif pooling == 'max':
            embedding = np.max(word_vectors, axis=0)
        elif pooling == 'sum':
            embedding = np.sum(word_vectors, axis=0)
        else:
            embedding = np.mean(word_vectors, axis=0)

        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_texts(self, texts: List[str], pooling: str = 'mean') -> np.ndarray:
        """
        嵌入多个文本

        Args:
            texts: 文本列表
            pooling: 池化策略
        Returns:
            嵌入向量矩阵
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text, pooling)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
        return float(similarity)

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        if self.word_vectors is not None:
            return len(self.word_vectors.key_to_index)
        return len(self.special_vectors)

class TextEmbeddingModel:
    """
    文本嵌入模型
    将文本描述转换为向量表示
    使用轻量级预训练词向量
    """

    def __init__(self,
                 model_name: str = 'glove-wiki-gigaword-100',
                 embedding_dim: int = 100,
                 device: Optional[str] = None):
        """
        初始化文本嵌入模型

        Args:
            model_name: 预训练模型名称
            embedding_dim: 嵌入维度
            device: 设备 (cuda/cpu)
        """
        self.embedding_dim = embedding_dim
        self.vocab = {}

        # 初始化轻量级嵌入模型
        try:
            self.lightweight_model = LightweightEmbeddingModel(
                model_name=model_name,
                embedding_dim=embedding_dim,
                device=device
            )
            print(f"使用轻量级嵌入模型: {model_name}")
            print(f"嵌入维度: {self.embedding_dim}")

        except Exception as e:
            print(f"加载轻量级模型失败，使用随机嵌入: {e}")
            self.lightweight_model = None
            self.embedding_dim = embedding_dim

    def create_simple_embedding(self, text: str) -> np.ndarray:
        """
        创建简单的文本嵌入（备用方法）
        当预训练模型不可用时使用
        """
        # 简化的嵌入生成，使用词频统计
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)

        for word in words:
            # 简单的哈希嵌入
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) % self.embedding_dim

            idx = self.vocab[word]
            embedding[idx] += 1

        # 归一化
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def embed_text(self, text: str, use_pretrained: bool = True) -> np.ndarray:
        """嵌入单个文本"""
        if use_pretrained and self.lightweight_model is not None:
            try:
                # 使用轻量级模型嵌入
                return self.lightweight_model.embed_text(text, pooling='mean')
            except Exception as e:
                print(f"预训练嵌入失败，使用简单嵌入: {e}")
                return self.create_simple_embedding(text)
        else:
            # 使用简单嵌入
            return self.create_simple_embedding(text)

    def embed_texts(self, texts: List[str], use_pretrained: bool = True) -> np.ndarray:
        """嵌入多个文本"""
        if use_pretrained and self.lightweight_model is not None and len(texts) > 0:
            return self.lightweight_model.embed_texts(texts)
        else:
            # 使用简单嵌入
            return np.array([self.create_simple_embedding(text) for text in texts])

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        if self.lightweight_model is not None:
            return self.lightweight_model.embedding_dim
        return self.embedding_dim

class MultiViewDataset(Dataset):
    """
    多视图数据集类
    处理Permission View, API View, URL & uses-feature View的嵌入
    """

    def __init__(self, prompt_results: Dict, labels: Dict, embedding_model: TextEmbeddingModel):
        """
        初始化数据集

        Args:
            prompt_results: 提示工程结果
            labels: 标签字典 {seq: label}
            embedding_model: 文本嵌入模型
        """
        self.embedding_model = embedding_model
        self.data = []
        self.labels = []

        self._prepare_data(prompt_results, labels)

    def _prepare_data(self, prompt_results: Dict, labels: Dict):
        """准备数据"""
        for seq, result in prompt_results.items():
            seq = int(seq)

            # 获取标签
            label = labels.get(seq, 0)  # 默认良性

            # 构建多视图文本
            view_texts = self._extract_view_texts(result)

            # 为每个视图生成嵌入
            view_embeddings = []
            for view_text in view_texts:
                embedding = self.embedding_model.embed_text(view_text)
                view_embeddings.append(embedding)

            # 拼接所有视图的嵌入
            if view_embeddings:
                # 确保所有嵌入维度相同
                emb_dim = self.embedding_model.get_embedding_dim()
                padded_embeddings = []
                for emb in view_embeddings:
                    if len(emb) < emb_dim:
                        # 填充
                        padded = np.zeros(emb_dim)
                        padded[:len(emb)] = emb
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb[:emb_dim])

                combined_embedding = np.concatenate(padded_embeddings)
                self.data.append(combined_embedding)
                self.labels.append(label)

    def _extract_view_texts(self, result: Dict) -> List[str]:
        """从结果中提取视图文本"""
        texts = []

        # Permission View文本
        permission_text = ""
        if 'view_summaries' in result and 'permission_summary' in result['view_summaries']:
            permission_text = result['view_summaries']['permission_summary']
        texts.append(permission_text)

        # API View文本
        api_text = ""
        if 'view_summaries' in result and 'api_summary' in result['view_summaries']:
            api_text = result['view_summaries']['api_summary']
        texts.append(api_text)

        # URL & uses-feature View文本
        url_text = ""
        if 'view_summaries' in result and 'url_uses_feature_summary' in result['view_summaries']:
            url_text = result['view_summaries']['url_uses_feature_summary']
        texts.append(url_text)

        # 添加功能描述文本
        function_text = ""
        if 'function_descriptions' in result:
            for view_type, descriptions in result['function_descriptions'].items():
                for feature, desc in descriptions.items():
                    function_text += f"{feature}: {desc}; "
        texts.append(function_text)

        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.FloatTensor(self.data[idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor

class DNNClassifier(nn.Module):
    """
    DNN分类器
    对应论文中使用的深度神经网络
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], num_classes: int = 2):
        """
        初始化DNN分类器

        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数量
        """
        super(DNNClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # 创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AppPoetDetector:
    """
    AppPoet检测器
    实现论文中的检测分类模块
    """

    def __init__(self,
                 embedding_dim: int = 100,
                 num_views: int = 4):
        """
        初始化检测器

        Args:
            embedding_dim: 每个视图的嵌入维度
            num_views: 视图数量
        """
        self.embedding_dim = embedding_dim
        self.num_views = num_views
        self.input_dim = embedding_dim * num_views

        # 使用更小的隐藏层以适应轻量级嵌入
        hidden_dims = [256, 128, 64]

        self.embedding_model = TextEmbeddingModel(
            model_name='glove-wiki-gigaword-100',
            embedding_dim=embedding_dim
        )
        self.classifier = DNNClassifier(self.input_dim, hidden_dims=hidden_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 移动模型到设备
        self.classifier.to(self.device)
        print(f"模型输入维度: {self.input_dim}")
        print(f"嵌入维度: {embedding_dim} (每个视图)")
        print(f"总输入维度: {self.input_dim}")

    def load_labels(self, conn, seqs) -> Dict:
        """加载标签"""
        labels = {}
        for seq in seqs:
            with conn.cursor() as cursor:
                sql = "SELECT label FROM app_label WHERE seq = %s"
                cursor.execute(sql, (seq,))
                result = cursor.fetchone()
                if result:
                    labels[seq] = 0 if result[0] == 'B' else 1
        return labels

    def train(self, prompt_results: Dict, labels: Dict,
              batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.001):
        """
        训练检测器

        Args:
            prompt_results: 提示工程结果
            labels: 标签字典
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
        """
        logger.info("开始训练AppPoet检测器...")
        logger.info(f"训练样本数: {len(prompt_results)}")
        logger.info(f"输入维度: {self.input_dim}")

        # 准备数据集
        dataset = MultiViewDataset(prompt_results, labels, self.embedding_model)

        if len(dataset) == 0:
            logger.error("数据集为空")
            return

        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

        # 训练循环
        best_val_accuracy = 0.0

        for epoch in range(epochs):
            # 训练阶段
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_accuracy = 100. * train_correct / train_total

            # 验证阶段
            val_accuracy, val_loss = self._validate(val_loader, criterion)
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self._save_model('apppoet_best_model.pth')

            logger.info(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss: {train_loss/len(train_loader):.4f}, "
                        f"Train Acc: {train_accuracy:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_accuracy:.2f}%")

        logger.info(f"训练完成，最佳验证准确率: {best_val_accuracy:.2f}%")

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证阶段"""
        self.classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.classifier(data)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_accuracy = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        return val_accuracy, avg_val_loss

    def evaluate(self, test_prompt_results: Dict, test_labels: Dict) -> Dict:
        """
        在测试集上评估模型

        Args:
            test_prompt_results: 测试集提示工程结果
            test_labels: 测试集标签
        Returns:
            评估指标字典
        """
        logger.info("在测试集上评估模型...")

        # 准备测试数据集
        test_dataset = MultiViewDataset(test_prompt_results, test_labels, self.embedding_model)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 评估
        self.classifier.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                outputs = self.classifier(data)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())

        # 计算指标
        from sklearn.metrics import balanced_accuracy_score
        balance_accuracy = balanced_accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)

        metrics = {
            'accuracy': balance_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"测试结果: "
                    f"Accuracy: {balance_accuracy:.4f}, "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"F1-Score: {f1:.4f}")

        return metrics

    def predict_single(self, prompt_result: Dict) -> Tuple[int, float]:
        """
        预测单个样本

        Args:
            prompt_result: 单个样本的提示工程结果
        Returns:
            (预测标签, 置信度)
        """
        self.classifier.eval()

        # 准备数据
        labels = {0: 0}  # 虚拟标签
        dataset = MultiViewDataset({0: prompt_result}, labels, self.embedding_model)

        if len(dataset) == 0:
            return 0, 0.0

        data, _ = dataset[0]
        data = data.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(data)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

            return predicted.item(), confidence.item()

    def _save_model(self, path: str):
        """保存模型"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'embedding_vocab': self.embedding_model.vocab,
            'embedding_dim': self.embedding_dim,
            'input_dim': self.input_dim
        }, path)
        logger.info(f"模型已保存到 {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.embedding_model.vocab = checkpoint['embedding_vocab']
        self.embedding_dim = checkpoint.get('embedding_dim', 100)
        self.input_dim = checkpoint.get('input_dim', 400)
        logger.info(f"模型已从 {path} 加载")

def main(TRAIN_PROMPT_FILE = "apppoet_prompt_results.json",
         MODEL_SAVE_PATH = "apppoet_detection_model.pth",
         TEST_PROMPT_FILE = "apppoet_test_prompt_results.json",
         test_file=r'/home/changxiaosong/python/malwareTest/test_0.8repartition.txt'):
    """主函数"""
    # 加载训练数据
    with open(TRAIN_PROMPT_FILE, 'r', encoding='utf-8') as f:
        train_prompt_results = json.load(f)

    logger.info(f"加载训练数据: {len(train_prompt_results)} 个样本")

    # 创建检测器 - 使用更小的嵌入维度
    detector = AppPoetDetector(embedding_dim=100)  # 使用100维GloVe嵌入

    # 加载标签
    conn = get_connection()
    seqs_train = [int(seq) for seq in train_prompt_results.keys()]
    labels = detector.load_labels(conn, seqs_train)
    conn.close()

    logger.info(f"加载标签: {len(labels)} 个样本有标签")

    # 训练模型
    detector.train(train_prompt_results, labels, epochs=30, learning_rate=0.001)

    # 保存模型
    detector._save_model(MODEL_SAVE_PATH)
    logger.info("AppPoet检测器训练完成")

    # 如果需要验证
    if TEST_PROMPT_FILE:
        logger.info("开始验证...")
        seqs_test = load_seqs_from_file(test_file)

        # 评估模型
        with open(TEST_PROMPT_FILE, 'r', encoding='utf-8') as f:
            test_prompt_results = json.load(f)

        # 加载测试标签
        conn = get_connection()
        test_labels = detector.load_labels(conn, seqs_test)
        conn.close()

        # 过滤出有标签的测试样本
        filtered_test_results = {}
        for seq in seqs_test:
            seq = int(seq)
            if seq in test_labels and str(seq) in test_prompt_results:
                filtered_test_results[str(seq)] = test_prompt_results[str(seq)]

        if filtered_test_results:
            metrics = detector.evaluate(filtered_test_results, test_labels)
            print('AppPoet_performance',metrics,flush=True)
            logger.info(f"验证完成，指标: {metrics}")
        else:
            logger.warning("没有可用的测试数据")

if __name__ == "__main__":
    TRAIN_PROMPT_FILE = "apppoet_prompt_results.json"
    MODEL_SAVE_PATH = "apppoet_detection_model.pth"
    TEST_PROMPT_FILE = TRAIN_PROMPT_FILE#"apppoet_test_prompt_results.json"  # 如果有测试集
    test_file = r'/home/changxiaosong/python/malwareTest/test_0.8repartition.txt'

    main(TRAIN_PROMPT_FILE, MODEL_SAVE_PATH, TEST_PROMPT_FILE, test_file)