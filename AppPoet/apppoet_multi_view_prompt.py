#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：apppoet_multi_view_prompt.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/1/15 10:22 
'''
# coding=utf-8
import platform
import sys

"""
AppPoet多视角提示工程模块
根据多视角特征获取大模型对APK的描述
对应论文第4.2节
"""

import json
import logging
import os
from typing import Dict, List, Tuple
from datetime import datetime
system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AppPoet")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")

from pr2_final.test001Method_new_9_4_3 import get_file_path_by_seq
from pr2_final.test001Method_new_9_4_3 import llm_chat
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class MultiViewPromptEngineer:
    """
    多视角提示工程师
    实现论文中的function description generation和view summary generation
    """

    def __init__(self,llm_model: str = "codellama:13b"):
        """
        初始化提示工程师

        Args:
            llm_model: LLM模型名称
        """
        self.llm_model = llm_model
        self.function_memory = {}  # 实现论文中的function memory

    def load_features(self, feature_file: str) -> Dict:
        """加载提取的特征"""
        try:
            with open(feature_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
            logger.info(f"成功加载特征文件: {feature_file}")
            return features
        except Exception as e:
            logger.error(f"加载特征文件失败: {e}")
            return {}

    def create_function_description_prompt(
            self,
            feature_type: str,
            feature_name: str,
            examples: List[Tuple] = None
    ) -> str:
        """
        创建功能描述提示模板
        对应论文Table 2和4.2.1节
        """

        type_descriptions = {
            'permission': 'Android permission',
            'api': 'Android API',
            'url': 'URL address',
            'uses_feature': 'hardware or software feature'
        }

        feature_type_desc = type_descriptions.get(feature_type, feature_type)

        # ===== LLM Prompt（英文）=====
        prompt = (
            f"You are an Android security expert with comprehensive knowledge of "
            f"{feature_type_desc}s and their functionalities. "
            f"Please describe the functionality of the following {feature_type_desc}."
        )

        # 上下文学习示例
        if examples:
            for example_feature, example_function in examples[:3]:
                prompt += f"\n{feature_type_desc}: {example_feature}"
                prompt += f"\nFunction: {example_function}"

        # 目标特征
        prompt += f"\n\n{feature_type_desc}: {feature_name}"
        prompt += "\nFunction: "

        return prompt

    def create_view_summary_prompt(
            self,
            view_type: str,
            feature_descriptions: Dict,
            package_name: str = "unknown"
    ) -> str:
        """
        创建视图总结提示模板
        对应论文Table 3和4.2.2节
        """

        view_descriptions = {
            'permission': 'a permission-based view',
            'api': 'a sensitive API usage-based view',
            'url_uses_feature': 'a URL and uses-feature based view'
        }

        feature_subtype_descriptions = {
            'permission': {
                'requested_permission': 'permissions declared in AndroidManifest.xml',
                'used_permission': 'permissions actually used in the source code'
            },
            'api': {
                'restricted_api': 'APIs that require specific permissions',
                'suspicious_api': 'other sensitive APIs related to accessing private data or system resources'
            },
            'url_uses_feature': {
                'url': 'URLs found in the source code',
                'uses_feature': 'hardware or software features declared in the manifest'
            }
        }

        view_desc = view_descriptions.get(view_type, view_type)
        feature_subtypes = feature_subtype_descriptions.get(view_type, {})

        # ===== LLM Prompt（英文）=====
        prompt = f"""
You are an Android security expert specializing in static analysis of Android applications.
Your task is to generate a behavior summary for the application "{package_name}"
from {view_desc}.

Task Description:
You must strictly follow the steps below to analyze the application and produce
a behavior summary from {view_desc}.

1. You are provided with the following {view_type}-related information.
Each item follows the format:
'{view_type} name: {view_type} function'.
"""

        for subtype, descriptions in feature_descriptions.items():
            subtype_desc = feature_subtypes.get(subtype, subtype)
            prompt += (
                f"\n1.1 {subtype} ({subtype_desc}): "
                f"{json.dumps(descriptions, ensure_ascii=False)}"
            )

        prompt += f"""

2. Based on the above information, perform a static analysis of the application
from {view_desc} and generate a concise behavior summary.

Output Requirements:
1. Summarize and interpret the known information from {view_desc}, focusing on
   high-risk {view_type} elements and their potential security implications.
2. Do not include any additional explanations or meta descriptions.
3. The output must be concise.
4. All statements must be strictly based on factual and known information.
5. Missing or empty items indicate the absence of such behavior.
6. Do not include speculation, recommendations, or subjective assumptions.
"""

        return prompt

    def generate_function_descriptions(self, features: Dict, seq: int) -> Dict:
        """
        生成功能描述
        实现论文中的function description generation
        """
        function_descriptions = {
            'permission': {},
            'api': {},
            'url_uses_feature': {}
        }

        try:
            seq_features = features.get(str(seq), {})
            if not seq_features:
                logger.warning(f"序列 {seq} 的特征为空")
                return function_descriptions
            task=[]






            permission_view = seq_features.get('permission_view', {})
            for permission in permission_view.get('requested_permissions', []):
                if permission not in self.function_memory:
                    prompt = self.create_function_description_prompt('permission', permission)
                    description = self._call_llm_for_function(prompt, permission)
                    self.function_memory[permission] = description
                function_descriptions['permission'][permission] = self.function_memory[permission]

            api_view = seq_features.get('api_view', {})
            for api in api_view.get('restricted_apis', []) + api_view.get('suspicious_apis', []):
                if api not in self.function_memory:
                    prompt = self.create_function_description_prompt('api', api)
                    description = self._call_llm_for_function(prompt, api)
                    self.function_memory[api] = description
                function_descriptions['api'][api] = self.function_memory[api]

            url_view = seq_features.get('url_uses_feature_view', {})
            for url in url_view.get('urls', []):
                if url not in self.function_memory:
                    prompt = self.create_function_description_prompt('url', url)
                    description = self._call_llm_for_function(prompt, url)
                    self.function_memory[url] = description
                function_descriptions['url_uses_feature'][url] = self.function_memory[url]

            for feature in url_view.get('uses_features', []):
                if feature not in self.function_memory:
                    prompt = self.create_function_description_prompt('uses_feature', feature)
                    description = self._call_llm_for_function(prompt, feature)
                    self.function_memory[feature] = description
                function_descriptions['url_uses_feature'][feature] = self.function_memory[feature]

            logger.info(f"为序列 {seq} 生成了功能描述")

        except Exception as e:
            logger.error(f"生成功能描述失败: {e}")

        return function_descriptions

    def generate_view_summaries(self, function_descriptions: Dict, seq: int) -> Dict:
        """
        生成视图摘要
        实现论文中的view summary generation
        """
        view_summaries = {}

        try:
            if function_descriptions.get('permission'):
                prompt = self.create_view_summary_prompt(
                    'permission',
                    {'requested_permission': function_descriptions['permission']},
                    f"app_{seq}"
                )
                view_summaries['permission_summary'] = self._call_llm_for_summary(prompt)

            if function_descriptions.get('api'):
                prompt = self.create_view_summary_prompt(
                    'api',
                    {'suspicious_api': function_descriptions['api']},
                    f"app_{seq}"
                )
                view_summaries['api_summary'] = self._call_llm_for_summary(prompt)

            if function_descriptions.get('url_uses_feature'):
                prompt = self.create_view_summary_prompt(
                    'url_uses_feature',
                    {'url': function_descriptions['url_uses_feature']},
                    f"app_{seq}"
                )
                view_summaries['url_uses_feature_summary'] = self._call_llm_for_summary(prompt)

            logger.info(f"为序列 {seq} 生成了视图摘要")

        except Exception as e:
            logger.error(f"生成视图摘要失败: {e}")

        return view_summaries

    def _call_llm_for_function(self, prompt: str, feature: str) -> str:
        """
        调用LLM生成功能描述
        使用llm_chat方法调用真实的LLM
        """
        try:
            # 创建对话列表
            talks = []

            # 构建英文提示词
            if 'permission' in prompt.lower():
                feature_type = "Android permission"
            elif 'api' in prompt.lower():
                feature_type = "Android API"
            elif 'url' in prompt.lower():
                feature_type = "URL"
            elif 'uses_feature' in prompt.lower():
                feature_type = "Android uses-feature"
            else:
                feature_type = "Android feature"

            # 系统提示
            system_prompt = f"""You are an Android security expert. Please describe the function of the following {feature_type}. 
    Keep your response concise and accurate."""

            # 用户提示
            user_prompt = f"{feature_type}: {feature}\nfunction:"

            # 添加示例（few-shot learning）
            if 'permission' in prompt.lower():
                examples = [
                    "permission: android.permission.CAMERA\nfunction: Allows access to the device camera for taking photos or recording videos",
                    "permission: android.permission.ACCESS_FINE_LOCATION\nfunction: Allows access to the device's precise location information",
                    "permission: android.permission.READ_SMS\nfunction: Allows reading SMS message content"
                ]
                user_prompt = "\n".join(examples) + "\n\n" + user_prompt
            elif 'api' in prompt.lower():
                examples = [
                    "API: Landroid/telephony/TelephonyManager;->getDeviceId\nfunction: Used to obtain the device's unique identifier (e.g., IMEI)",
                    "API: Landroid/location/LocationManager;->getLastKnownLocation\nfunction: Used to obtain the device's last known location information",
                    "API: Landroid/hardware/Camera;->open\nfunction: Used to open the device camera for taking photos or recording videos"
                ]
                user_prompt = "\n".join(examples) + "\n\n" + user_prompt
            elif 'url' in prompt.lower():
                examples = [
                    "URL: http://example.com\nfunction: Network address the application connects to, potentially used for data communication",
                    "URL: https://api.example.com\nfunction: Secure network address used for API calls and data transmission"
                ]
                user_prompt = "\n".join(examples) + "\n\n" + user_prompt

# 构建完整的对话
            talks.append({"role": "user", "content": system_prompt})
            talks.append({"role": "assistant", "content": "I understand. I will provide concise and accurate descriptions of Android features."})
            talks.append({"role": "user", "content": user_prompt})

            # 调用LLM
            task = "Please provide only the function description without any additional text."
            talks, llm_response = llm_chat(0, talks.copy(), task, self.llm_model)
            # 清理响应
            response = llm_response.strip()

            # 如果响应包含"function:"，提取后面的内容
            if "function:" in response:
                response = response.split("function:", 1)[1].strip()

            # 移除可能的引号和多余空格
            response = response.strip('"').strip("'").strip()

            # 验证响应是否合理
            # if len(response) < 5 or len(response) > 200:
            #     logger.warning(f"LLM返回的描述长度异常: {response}")

            # logger.info(f"LLM返回 {feature_type} '{feature}' 的描述: {response}")
            return response

        except ImportError as e:
            logger.error(f"无法导入llm_chat: {e}")
        except Exception as e:
            logger.error(f"调用LLM失败: {e}")
    def _call_llm_for_summary(self, prompt: str) -> str:
        """
            调用LLM生成视图摘要
            使用llm_chat方法调用真实的LLM
        """
        try:
            # 创建对话列表
            talks = []

            # 解析视图类型
            view_type = "unknown"
            if 'permission' in prompt.lower():
                view_type = "Permission View"
            elif 'api' in prompt.lower():
                view_type = "API View"
            elif 'url' in prompt.lower() or 'uses-feature' in prompt.lower():
                view_type = "URL & Uses-Feature View"

            # 系统提示
            system_prompt = f"""You are an expert Android security analyst. Analyze the {view_type} of an Android application and provide a behavior summary.

Requirements:
1. Analyze from the perspective of {view_type}
2. Focus on high-risk features and their potential risks
3. Provide objective analysis strictly based on facts
4. Do not include subjective assumptions or suggestions
5. Output must be concise

Output format:
Behavior Summary: [Your analysis here]"""

            # 提取用户输入的关键信息
            # 查找<Task Description>和</Task Description>之间的内容
            import re
            task_match = re.search(r'<Task Description>:(.*?)<Output Description and Requirements>', prompt, re.DOTALL)
            if task_match:
                task_content = task_match.group(1).strip()

                # 提取具体内容
                feature_content_match = re.search(r'1\.1-\s*(.*?)\s*1\.2-', task_content, re.DOTALL)
                if feature_content_match:
                    feature_content = feature_content_match.group(1)
                else:
                    feature_content = task_content
            else:
                feature_content = prompt

            # 清理内容，移除多余的格式
            feature_content = feature_content.replace("'", "").replace('"', '').strip()

            # 构建用户提示
            user_prompt = f"""Please analyze the following information from {view_type} and provide a behavior summary.

Application Information:
{feature_content}

Behavior Summary:"""

            # 构建完整的对话
            talks.append({"role": "user", "content": system_prompt})
            talks.append({"role": "assistant", "content": "I understand. I will analyze the Android application's behavior from the specified view and provide a concise summary focusing on potential risks."})
            talks.append({"role": "user", "content": user_prompt})

            # 调用LLM
            task = "Provide only the behavior summary without any additional text, explanations, or suggestions."
            talks, llm_response = llm_chat(0, talks.copy(), task, self.llm_model)
            # 清理响应
            response = llm_response.strip()

            # 提取"Behavior Summary:"之后的内容
            if "Behavior Summary:" in response:
                response = response.split("Behavior Summary:", 1)[1].strip()
            elif "行为摘要:" in response:
                response = response.split("行为摘要:", 1)[1].strip()
            elif "summary:" in response.lower():
                # 查找所有可能的summary关键词
                import re
                summary_match = re.search(r'(?:summary|summary:|行为分析|行为分析:)\s*(.*)', response, re.IGNORECASE | re.DOTALL)
                if summary_match:
                    response = summary_match.group(1).strip()

            # 移除可能的格式标记和多余内容
            response = re.sub(r'^\s*(?:[\-\*•])\s*', '', response)  # 移除项目符号
            response = re.sub(r'\s*\n\s*\n\s*', '\n', response)  # 压缩空行
            response = response.strip()

            # 验证响应是否合理
            if len(response) < 20:
                logger.warning(f"LLM返回的摘要过短: {response}")
                return self._get_fallback_summary(view_type)

            # 检查是否包含不必要的内容
            forbidden_phrases = [
                "further analysis", "dynamic analysis", "additional information",
                "may be", "should", "recommend"
            ]

            for phrase in forbidden_phrases:
                if phrase in response.lower():
                    response = response.replace(phrase, "")
                    logger.info(f"移除禁止短语: {phrase}")

            logger.info(f"LLM返回 {view_type} 摘要 (长度: {len(response)}): {response[:100]}...")
            return response

        except ImportError as e:
            logger.error(f"无法导入llm_chat: {e}")
        except Exception as e:
            logger.error(f"调用LLM生成摘要失败: {e}")
    def process_sequence(self, features: Dict, seq: int) -> Dict:
        """
        处理单个序列，生成所有描述和摘要
        """
        result = {
            'seq': seq,
            'function_descriptions': {},
            'view_summaries': {},
            'timestamp': datetime.now().isoformat()
        }

        # 生成功能描述
        function_descriptions = self.generate_function_descriptions(features, seq)
        result['function_descriptions'] = function_descriptions

        # 生成视图摘要
        view_summaries = self.generate_view_summaries(function_descriptions, seq)
        result['view_summaries'] = view_summaries

        return result

    def batch_process_sequences(self, feature_file: str, seqs: List[int], output_file: str = 'apppoet_prompt_results.json'):
        """
        批量处理序列

        Args:
            feature_file: 特征文件路径
            seqs: 要处理的序列列表
            output_file: 输出文件
        """
        # 加载特征
        features = self.load_features(feature_file)
        if not features:
            return

        results = {}

        for i, seq in enumerate(seqs):
            logger.info(f"正在处理序列 {seq} ({i+1}/{len(seqs)})")

            result = self.process_sequence(features, seq)
            results[seq] = result

            # 每处理5个序列保存一次
            if (i + 1) % 5 == 0:
                self._save_results_intermediate(results, output_file)

        # 保存最终结果
        self._save_results_intermediate(results, output_file)
        logger.info(f"提示工程处理完成，结果保存到 {output_file}")

    def _save_results_intermediate(self, results: Dict, output_file: str):
        """保存中间结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存结果文件失败: {e}")

def main(FEATURE_FILE = "apppoet_extracted_features.json",
    TRAIN_FILE = "train_0.8repartition_little.txt",
OUTPUT_FILE = "apppoet_prompt_results.json"
):
    # 加载训练序列
    from apppoet_feature_extractor import AppPoetFeatureExtractor
    extractor = AppPoetFeatureExtractor()
    seqs = extractor.load_seqs_from_file(TRAIN_FILE)

    if not seqs:
        logger.error("未找到有效的序列号")
        return

    # 创建提示工程师
    prompt_engineer = MultiViewPromptEngineer("codellama:13b")

    # 批量处理序列
    prompt_engineer.batch_process_sequences(FEATURE_FILE, seqs, OUTPUT_FILE)

if __name__ == "__main__":
    # 配置参数
    FEATURE_FILE = "apppoet_extracted_features.json"
    TRAIN_FILE = "train_0.8repartition_little.txt"
    OUTPUT_FILE = "apppoet_prompt_results.json"
    main(FEATURE_FILE,TRAIN_FILE,OUTPUT_FILE)