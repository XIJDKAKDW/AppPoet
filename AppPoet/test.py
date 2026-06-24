#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/1/15 12:15 
'''
import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from typing import List, Optional

class CodeBERTEmbeddingModel:
    """
    使用CodeBERT模型进行文本嵌入
    论文中使用的text-embedding-ada-002 (1536维) 替换为 CodeBERT (768维)
    """

    def __init__(self, model_name: str = "microsoft/codebert-base",
                 max_length: int = 512,
                 device: Optional[str] = None):
        """
        初始化CodeBERT嵌入模型

        Args:
            model_name: CodeBERT模型名称
            max_length: 最大输入长度
            device: 设备 (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 初始化tokenizer和模型
        self.tokenizer = None
        self.model = None
        self._initialize_model()

        # CodeBERT的输出维度是768
        self.embedding_dim = 768

    def _initialize_model(self):
        """初始化CodeBERT模型"""
        try:
            from transformers import logging
            logging.set_verbosity_error()

            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式

            print(f"成功加载CodeBERT模型: {self.model_name}")
            print(f"模型设备: {self.device}")
            print(f"嵌入维度: {self.embedding_dim}")

        except Exception as e:
            print(f"加载CodeBERT模型失败: {e}")
            raise

    def embed_text(self, text: str, pooling_strategy: str = 'mean') -> np.ndarray:
        """
        嵌入单个文本

        Args:
            text: 输入文本
            pooling_strategy: 池化策略 ('mean', 'cls', 'max')
        Returns:
            嵌入向量 (768维)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未初始化")

        # 对文本进行编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 提取嵌入
        embeddings = self._pool_embeddings(last_hidden_state, pooling_strategy)

        # 转换为numpy数组
        embeddings_np = embeddings.cpu().numpy()

        # 归一化
        norm = np.linalg.norm(embeddings_np)
        if norm > 0:
            embeddings_np = embeddings_np / norm

        return embeddings_np

    def embed_texts(self, texts: List[str], batch_size: int = 32,
                    pooling_strategy: str = 'mean') -> np.ndarray:
        """
        嵌入多个文本

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            pooling_strategy: 池化策略
        Returns:
            嵌入向量矩阵 [num_texts, 768]
        """
        if not texts:
            return np.array([])

        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            try:
                # 对批次进行编码
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                )

                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 前向传播
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    last_hidden_state = outputs.last_hidden_state

                # 提取嵌入
                batch_embeddings = self._pool_embeddings(last_hidden_state, pooling_strategy)

                # 转换为numpy并归一化
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                for j in range(batch_embeddings_np.shape[0]):
                    norm = np.linalg.norm(batch_embeddings_np[j])
                    if norm > 0:
                        batch_embeddings_np[j] = batch_embeddings_np[j] / norm

                all_embeddings.append(batch_embeddings_np)

            except Exception as e:
                print(f"处理批次 {i//batch_size} 失败: {e}")
                # 使用零向量作为占位符
                placeholder = np.zeros((len(batch_texts), self.embedding_dim))
                all_embeddings.append(placeholder)

        # 合并所有批次
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.zeros((0, self.embedding_dim))

    def _pool_embeddings(self, hidden_states: torch.Tensor, strategy: str = 'mean') -> torch.Tensor:
        """
        池化隐藏状态

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            strategy: 池化策略
        Returns:
            池化后的嵌入 [batch_size, hidden_size]
        """
        if strategy == 'mean':
            # 平均池化 (排除padding tokens)
            attention_mask = hidden_states != self.tokenizer.pad_token_id
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif strategy == 'cls':
            # 使用[CLS] token
            return hidden_states[:, 0, :]

        elif strategy == 'max':
            # 最大池化
            return torch.max(hidden_states, dim=1)[0]

        else:
            raise ValueError(f"不支持的池化策略: {strategy}")

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度

        Args:
            text1: 文本1
            text2: 文本2
        Returns:
            余弦相似度
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
        return float(similarity)

    def cleanup(self):
        """清理模型资源"""
        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 修改原始的TextEmbeddingModel类
class TextEmbeddingModel:
    """
    文本嵌入模型
    将文本描述转换为向量表示
    使用CodeBERT模型替代text-embedding-ada-002
    """

    def __init__(self, model_name: str = "microsoft/codebert-base",
                 embedding_dim: int = 768,
                 device: Optional[str] = None):
        """
        初始化文本嵌入模型

        Args:
            model_name: CodeBERT模型名称
            embedding_dim: 嵌入维度 (CodeBERT默认768维)
            device: 设备 (cuda/cpu)
        """
        self.embedding_dim = embedding_dim
        self.vocab = {}

        # 初始化CodeBERT模型
        try:
            self.codebert = CodeBERTEmbeddingModel(
                model_name=model_name,
                device=device
            )
            print(f"使用CodeBERT模型: {model_name}")

        except Exception as e:
            print(f"加载CodeBERT失败，使用简单嵌入: {e}")
            self.codebert = None

    def create_simple_embedding(self, text: str) -> np.ndarray:
        """
        创建简单的文本嵌入（备用方法）
        当CodeBERT不可用时使用
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

    def embed_text(self, text: str, use_codebert: bool = True) -> np.ndarray:
        """嵌入单个文本"""
        if use_codebert and self.codebert is not None:
            try:
                # 使用CodeBERT嵌入
                return self.codebert.embed_text(text, pooling_strategy='mean')
            except Exception as e:
                print(f"CodeBERT嵌入失败，使用简单嵌入: {e}")
                return self.create_simple_embedding(text)
        else:
            # 使用简单嵌入
            return self.create_simple_embedding(text)

    def embed_texts(self, texts: List[str], batch_size: int = 32,
                    use_codebert: bool = True) -> np.ndarray:
        """嵌入多个文本"""
        if use_codebert and self.codebert is not None and len(texts) > 0:
            try:
                # 使用CodeBERT批量嵌入
                return self.codebert.embed_texts(texts, batch_size=batch_size)
            except Exception as e:
                print(f"CodeBERT批量嵌入失败，使用简单嵌入: {e}")
                return np.array([self.create_simple_embedding(text) for text in texts])
        else:
            # 使用简单嵌入
            return np.array([self.create_simple_embedding(text) for text in texts])


# 使用示例
if __name__ == "__main__":
    # 测试CodeBERT嵌入
    embedder = TextEmbeddingModel()

    # 测试单个文本
    text1 = "android.permission.CAMERA allows access to device camera"
    embedding1 = embedder.embed_text(text1)
    print(f"文本1嵌入形状: {embedding1.shape}")
    print(f"文本1嵌入前10维: {embedding1[:10]}")

    # 测试多个文本
    texts = [
        "访问摄像头权限",
        "获取位置信息API",
        "读取短信内容",
        "访问外部存储"
    ]
    embeddings = embedder.embed_texts(texts)
    print(f"批量嵌入形状: {embeddings.shape}")

    # 清理资源
    if embedder.codebert:
        embedder.codebert.cleanup()