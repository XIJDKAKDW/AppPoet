#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：apppoet_diagnostic_report.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/1/15 10:23 
'''
# coding=utf-8
import platform
import sys

"""
AppPoet诊断报告生成器模块
根据多视角特征输入LLM生成恶意行为报告
对应论文第4.4节和附录B
"""

import json
import logging
import os
from typing import Dict, List
from datetime import datetime
system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AppPoet")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")


from apppoet_feature_extractor import AppPoetFeatureExtractor
# 设置日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class DiagnosticReportGenerator:
    """
    诊断报告生成器
    实现论文中的diagnostic report generation
    """

    def __init__(self, llm_api_url: str = None, llm_model: str = "gpt-4"):
        """
        初始化报告生成器

        Args:
            llm_api_url: LLM API地址
            llm_model: LLM模型名称
        """
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model

    def create_diagnostic_prompt(self, seq: int, features: Dict,
                                 function_descriptions: Dict, view_summaries: Dict,
                                 detection_result: Dict) -> str:
        """
        创建诊断报告提示
        对应论文Table 4和附录B
        """
        package_name = f"com.example.app_{seq}"
        classification = "恶意" if detection_result.get('prediction', 0) == 1 else "良性"
        confidence = detection_result.get('confidence', 0.5)

        prompt = f"""你是一名Android安全领域的专家，专门通过静态分析审核Android应用程序。你的任务是结合已知信息和你的专业知识，为给定的Android应用程序生成诊断报告。

任务描述：
你必须严格遵循以下步骤分析名为"{package_name}"的应用程序，并输出诊断报告：

1. 首先，你应该知道该应用程序被分类器分类为【{classification}】（置信度: {confidence:.2f}）。

2. 然后，你获得不同视图下的描述和摘要如下：

2.1 <Permission View>
2.1.1 <requested permission>: {json.dumps(function_descriptions.get('permission', {}), ensure_ascii=False)}
2.1.2 <used permission>: {json.dumps({}, ensure_ascii=False)}  # 简化处理
2.1.3 <permission view summary>: {view_summaries.get('permission_summary', '')}

2.2 <API View>
2.2.1 <restricted API>: {json.dumps({k: v for k, v in function_descriptions.get('api', {}).items() if 'restricted' in k.lower()}, ensure_ascii=False)}
2.2.2 <suspicious API>: {json.dumps({k: v for k, v in function_descriptions.get('api', {}).items() if 'suspicious' in k.lower() or 'restricted' not in k.lower()}, ensure_ascii=False)}
2.2.3 <API view summary>: {view_summaries.get('api_summary', '')}

2.3 <URL & uses-feature View>
2.3.1 <uses-feature>: {json.dumps({k: v for k, v in function_descriptions.get('url_uses_feature', {}).items() if 'http' not in k.lower() and 'www' not in k.lower()}, ensure_ascii=False)}
2.3.2 <URL>: {json.dumps({k: v for k, v in function_descriptions.get('url_uses_feature', {}).items() if 'http' in k.lower() or 'www' in k.lower()}, ensure_ascii=False)}
2.3.3 <URL & uses-feature view summary>: {view_summaries.get('url_uses_feature_summary', '')}

3. 现在你不仅知道应用程序是否恶意，还获得了不同视图的特征功能描述和视图行为摘要。你应该基于上述信息开始静态分析，并根据<输出描述和要求>为应用程序生成诊断报告。

输出描述和要求：
1. 你的诊断报告应基于上述信息，重点关注其潜在风险行为。
2. 你的报告应包含一个总结，以要点形式描述所有可能的潜在风险。总结必须考虑所有视图中的恶意行为。每个潜在恶意行为的要点需要指出具体的风险点，例如某个特性、API等。
3. 你的报告应基于总结的潜在恶意行为，为后续检测提供详细的指导步骤。

术语解释：
1. Permission View: 基于应用权限的视角
2. API View: 基于应用中使用敏感API的视角
3. URL & uses-feature View: 基于声明的uses-feature和代码中URL的视角

请严格按照以下格式输出诊断报告：
========================================
诊断报告 - {package_name}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

1. 应用概览
   - 应用名称: {package_name}
   - 检测结果: {classification}
   - 置信度: {confidence:.2f}

2. 潜在风险总结
   [以要点形式列出所有潜在风险]

3. 详细分析
   [对每个风险点的详细分析]

4. 后续检测建议
   [基于分析结果的具体建议]

========================================
报告结束
========================================
"""

        return prompt

    def _call_llm_for_diagnostic(self, prompt: str) -> str:
        """
        调用LLM生成诊断报告
        简化实现
        """
        # 在实际应用中，应该调用真正的LLM API
        # 这里提供一个示例报告

        if "恶意" in prompt:
            report = """========================================
诊断报告 - com.example.app_89738
生成时间: 2024-01-15 10:30:00
========================================

1. 应用概览
   - 应用名称: com.example.app_89738
   - 检测结果: 恶意
   - 置信度: 0.92

2. 潜在风险总结
   - 未经授权的位置跟踪：应用使用ACCESS_FINE_LOCATION权限和GPS硬件功能
   - 隐私侵犯：通过摄像头和麦克风权限可能进行监控
   - 敏感设备信息访问：使用TelephonyManager.getDeviceId等API
   - 未经授权的数据传输：存在可疑URL和HTTP POST API调用
   - 外部存储操作：具有写入外部存储的权限

3. 详细分析
   3.1 位置跟踪风险
       - 使用的权限：android.permission.ACCESS_FINE_LOCATION
       - 使用的硬件：android.hardware.location.gps
       - 风险：可能持续跟踪用户位置，用于行为分析或位置监控

   3.2 隐私侵犯风险
       - 使用的权限：android.permission.CAMERA, android.permission.RECORD_AUDIO
       - 使用的API：android.media.MediaRecorder相关调用
       - 风险：可能在用户不知情的情况下拍照、录像或录音

   3.3 信息泄露风险
       - 敏感API：Landroid/telephony/TelephonyManager.getDeviceId
       - 使用的权限：android.permission.READ_PHONE_STATE
       - 风险：获取设备唯一标识符，可能用于用户追踪或设备指纹识别

   3.4 数据外传风险
       - 可疑URL：包含多个外部服务器地址
       - 使用的API：Lorg/apache/http/client/methods/HttpPost
       - 风险：可能将收集的数据传输到远程服务器

4. 后续检测建议
   1) 代码审查：重点审查敏感API的使用上下文
   2) 网络流量分析：监控应用与外部服务器的通信
   3) 动态行为分析：在沙箱环境中运行应用，观察实际行为
   4) 存储访问监控：检查应用对存储的读写操作
   5) 用户同意验证：确认敏感操作是否获得用户明确同意

========================================
报告结束
========================================
"""
        else:
            report = """========================================
诊断报告 - com.example.app_12345
生成时间: 2024-01-15 10:30:00
========================================

1. 应用概览
   - 应用名称: com.example.app_12345
   - 检测结果: 良性
   - 置信度: 0.85

2. 潜在风险总结
   - 无重大安全风险发现
   - 应用请求的权限与功能基本匹配
   - API使用符合正常应用模式

3. 详细分析
   3.1 权限使用分析
       - 使用的权限均为常见应用功能所需
       - 权限与声明的功能一致
       - 无过度权限请求

   3.2 API使用分析
       - 使用的API均为标准Android API
       - 无敏感API的异常使用模式
       - API调用与功能需求匹配

   3.3 网络行为分析
       - 连接的URL均为可信服务
       - 无可疑网络行为模式

4. 后续检测建议
   1) 常规安全监控：持续观察应用更新
   2) 权限使用审计：定期检查权限使用情况
   3) 网络行为监控：确保网络连接符合隐私政策

========================================
报告结束
========================================
"""

        return report

    def generate_diagnostic_report(self, seq: int,
                                   features: Dict,
                                   function_descriptions: Dict,
                                   view_summaries: Dict,
                                   detection_result: Dict) -> Dict:
        """
        生成单个应用的诊断报告

        Args:
            seq: 序列号
            features: 特征数据
            function_descriptions: 功能描述
            view_summaries: 视图摘要
            detection_result: 检测结果
        Returns:
            诊断报告字典
        """
        logger.info(f"为序列 {seq} 生成诊断报告")

        # 创建提示
        prompt = self.create_diagnostic_prompt(
            seq, features, function_descriptions, view_summaries, detection_result
        )

        # 调用LLM生成报告
        report_text = self._call_llm_for_diagnostic(prompt)

        # 解析报告
        parsed_report = self._parse_diagnostic_report(report_text, seq)

        # 添加元数据
        parsed_report['metadata'] = {
            'seq': seq,
            'generation_time': datetime.now().isoformat(),
            'model_used': self.llm_model,
            'detection_result': detection_result
        }

        return parsed_report

    def _parse_diagnostic_report(self, report_text: str, seq: int) -> Dict:
        """
        解析诊断报告文本为结构化数据
        """
        parsed = {
            'seq': seq,
            'report_text': report_text,
            'sections': {}
        }

        try:
            lines = report_text.split('\n')
            current_section = None
            section_content = []

            for line in lines:
                line = line.strip()

                if line.startswith('1. 应用概览'):
                    current_section = 'overview'
                    section_content = []
                elif line.startswith('2. 潜在风险总结'):
                    if current_section:
                        parsed['sections'][current_section] = '\n'.join(section_content)
                    current_section = 'risk_summary'
                    section_content = []
                elif line.startswith('3. 详细分析'):
                    if current_section:
                        parsed['sections'][current_section] = '\n'.join(section_content)
                    current_section = 'detailed_analysis'
                    section_content = []
                elif line.startswith('4. 后续检测建议'):
                    if current_section:
                        parsed['sections'][current_section] = '\n'.join(section_content)
                    current_section = 'recommendations'
                    section_content = []
                elif line.startswith('========================================'):
                    continue
                elif current_section and line:
                    section_content.append(line)

            # 保存最后一个部分
            if current_section and section_content:
                parsed['sections'][current_section] = '\n'.join(section_content)

        except Exception as e:
            logger.error(f"解析诊断报告失败: {e}")
            parsed['sections'] = {'raw': report_text}

        return parsed

    def generate_batch_reports(self, seqs: List[int],
                               prompt_results_file: str,
                               detection_results_file: str,
                               features_file: str,
                               output_dir: str = "diagnostic_reports"):
        """
        批量生成诊断报告

        Args:
            seqs: 序列号列表
            prompt_results_file: 提示工程结果文件
            detection_results_file: 检测结果文件
            features_file: 特征文件
            output_dir: 输出目录
        """
        logger.info(f"开始为 {len(seqs)} 个序列生成诊断报告")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        with open(prompt_results_file, 'r', encoding='utf-8') as f:
            prompt_results = json.load(f)

        with open(detection_results_file, 'r', encoding='utf-8') as f:
            detection_results = json.load(f)

        with open(features_file, 'r', encoding='utf-8') as f:
            features = json.load(f)

        all_reports = {}

        for i, seq in enumerate(seqs):
            seq_str = str(seq)
            logger.info(f"正在处理序列 {seq} ({i+1}/{len(seqs)})")

            # 获取数据
            seq_prompt_result = prompt_results.get(seq_str, {})
            seq_detection_result = detection_results.get(seq_str, {})
            seq_features = features.get(seq_str, {})

            if not seq_prompt_result or not seq_detection_result:
                logger.warning(f"序列 {seq} 的数据不完整，跳过")
                continue

            # 生成报告
            report = self.generate_diagnostic_report(
                seq,
                seq_features,
                seq_prompt_result.get('function_descriptions', {}),
                seq_prompt_result.get('view_summaries', {}),
                seq_detection_result
            )

            all_reports[seq] = report

            # 保存单个报告
            report_file = os.path.join(output_dir, f"seq_{seq}_diagnostic_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # 保存文本报告
            text_report_file = os.path.join(output_dir, f"seq_{seq}_report.txt")
            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write(report['report_text'])

        # 保存所有报告
        combined_file = os.path.join(output_dir, "all_diagnostic_reports.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)

        # 生成摘要
        self._generate_summary_statistics(all_reports, output_dir)

        logger.info(f"诊断报告生成完成，保存在 {output_dir}")

    def _generate_summary_statistics(self, reports: Dict, output_dir: str):
        """生成摘要统计"""
        total_reports = len(reports)
        malicious_count = 0
        benign_count = 0

        risk_categories = {}

        for seq, report in reports.items():
            # 统计分类
            if "恶意" in report.get('report_text', ''):
                malicious_count += 1
            elif "良性" in report.get('report_text', ''):
                benign_count += 1

            # 提取风险关键词
            report_text = report.get('report_text', '').lower()

            risk_keywords = [
                '位置', '摄像头', '麦克风', '隐私', '追踪',
                '信息泄露', '数据外传', '存储', '网络', 'api'
            ]

            for keyword in risk_keywords:
                if keyword in report_text:
                    risk_categories[keyword] = risk_categories.get(keyword, 0) + 1

        summary = {
            'total_reports': total_reports,
            'malicious_count': malicious_count,
            'benign_count': benign_count,
            'malicious_rate': malicious_count / total_reports if total_reports > 0 else 0,
            'risk_category_distribution': risk_categories,
            'generation_time': datetime.now().isoformat()
        }

        summary_file = os.path.join(output_dir, "diagnostic_reports_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印摘要
        print("\n=== 诊断报告摘要 ===")
        print(f"总报告数: {total_reports}")
        print(f"恶意应用: {malicious_count}")
        print(f"良性应用: {benign_count}")
        print(f"恶意率: {summary['malicious_rate']:.2%}")
        print("\n风险类别分布:")
        for category, count in risk_categories.items():
            print(f"  {category}: {count}")

def main(PROMPT_RESULTS_FILE = "apppoet_prompt_results.json",
    DETECTION_RESULTS_FILE = "apppoet_prompt_results.json",
FEATURES_FILE = "apppoet_extracted_features.json",
OUTPUT_DIR = "apppoet_diagnostic_reports",
test_file = r'/home/changxiaosong/python/malwareTest/test_0.8repartition.txt'):
    """主函数"""
    # 配置参数

    # 选择一些序列生成报告（例如前10个）
    extractor = AppPoetFeatureExtractor()
    seqs = extractor.load_seqs_from_file(test_file)

    if not seqs:
        logger.error("未找到有效的序列号")
        return

    # 创建报告生成器
    report_generator = DiagnosticReportGenerator()

    # 批量生成报告
    report_generator.generate_batch_reports(
        seqs,
        PROMPT_RESULTS_FILE,
        DETECTION_RESULTS_FILE,
        FEATURES_FILE,
        OUTPUT_DIR
    )

if __name__ == "__main__":
    PROMPT_RESULTS_FILE = "apppoet_prompt_results.json",
    DETECTION_RESULTS_FILE = "apppoet_prompt_results.json",
    FEATURES_FILE = "apppoet_extracted_features.json",
    OUTPUT_DIR = "apppoet_diagnostic_reports",
    test_file = r'/home/changxiaosong/python/malwareTest/test_0.8repartition.txt'
    main(PROMPT_RESULTS_FILE,DETECTION_RESULTS_FILE,FEATURES_FILE,OUTPUT_DIR,test_file)