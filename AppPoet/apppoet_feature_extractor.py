#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：malwareTest 
@File    ：apppoet_feature_extractor.py
@IDE     ：PyCharm 
@Author  ：常晓松
@Date    ：2026/1/15 10:11 
'''
# coding=utf-8
import sys

"""
AppPoet特征提取模块
静态分析提取权限、API和用户层特征
对应论文第4.1节
"""

import os
import json
import logging
import platform
import subprocess
from typing import Dict, List
import xml.etree.ElementTree as ET
import re

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AppPoet")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")

from pr2_new_2.test001Method_new_9_4_2 import get_file_path_by_seq

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
class AppPoetFeatureExtractor:
    """
    AppPoet特征提取器
    实现论文中Permission View, API View, URL & uses-feature View的特征提取
    """

    def __init__(self, ):
        """
        初始化特征提取器

        """
        self.system = platform.system()

    def load_seqs_from_file(self, file_path: str) -> List[int]:
        """从文件加载序列号"""
        seqs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line.isdigit():
                        seqs.append(int(line))
            logger.info(f"从 {file_path} 加载了 {len(seqs)} 个序列号")
        except Exception as e:
            logger.error(f"加载文件时出错: {e}")
        return seqs

    def get_apk_path_by_seq(self, seq: int) -> str:
        """根据序列号获取APK路径"""
        # 这里假设APK文件的命名规则，可以根据实际情况修改
        apkName,apkPath=get_file_path_by_seq(seq)
        if system == "Linux":
            apkPath=apkPath.replace('F:/malware-app','/home/changxiaosong/dataset')
        else:
            apkPath=apkPath.replace('F:/malware-app','E:/malware-app')

        if not os.path.exists(apkPath):
            logger.warning(f"APK文件不存在: {apkPath}")
            return None

        return apkPath

    def decompile_apk(self, apk_path: str, output_dir: str) -> Dict:
        """
        反编译APK文件
        Args:
            apk_path: APK文件路径
            output_dir: 反编译输出目录
        Returns:
            反编译结果字典
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            # 使用apktool反编译
            cmd = ['apktool', 'd', apk_path, '-o', output_dir, '-f']
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"反编译失败: {result.stderr}")
                return None

            # 解析反编译结果
            decompiled_files = {
                'manifest': os.path.join(output_dir, 'AndroidManifest.xml'),
                'smali_dir': os.path.join(output_dir, 'smali'),
                'res_dir': os.path.join(output_dir, 'res'),
                'assets_dir': os.path.join(output_dir, 'assets')
            }

            logger.info(f"APK反编译成功: {apk_path}")
            return decompiled_files

        except Exception as e:
            logger.error(f"反编译过程中出错: {e}")
            return None

    def extract_permission_view(self, manifest_file: str, smali_dir: str = None) -> Dict:
        """
            提取Permission View特征
            对应论文Table 1: requested permission和used permission

            Args:
                manifest_file: AndroidManifest.xml文件路径
                smali_dir: smali代码目录路径（可选）
        """
        permissions = {
            'requested_permissions': [],
            'used_permissions': []
        }

        try:
            # 解析AndroidManifest.xml
            tree = ET.parse(manifest_file)
            root = tree.getroot()

            # 提取声明权限 (requested permissions)
            for uses_permission in root.findall('.//uses-permission'):
                android_name = uses_permission.get('{http://schemas.android.com/apk/res/android}name')
                if android_name:
                    permissions['requested_permissions'].append(android_name)

            # 提取实际使用权限 - 分析smali代码中的权限检查
            if smali_dir and os.path.exists(smali_dir):
                used_permissions = self._analyze_permission_usage_in_smali(smali_dir)
                permissions['used_permissions'] = used_permissions
            else:
                logger.warning("smali目录不存在，无法分析实际使用权限,请检查")
                # 如果没有smali目录，使用声明权限作为近似
                permissions['used_permissions'] = permissions['requested_permissions'].copy()

            logger.info(f"提取到 {len(permissions['requested_permissions'])} 个声明权限和 {len(permissions['used_permissions'])} 个使用权限")

        except Exception as e:
            logger.error(f"解析manifest文件失败: {e}")

        return permissions



    def _analyze_permission_usage_in_smali(self, smali_dir: str) -> List[str]:
        """
        分析smali代码中的权限使用
        通过查找权限检查相关的API调用来确定实际使用的权限

        Args:
            smali_dir: smali代码目录路径
        Returns:
            实际使用的权限列表
        """
        used_permissions = set()

        # 定义权限检查相关的API模式 - 使用smali格式
        permission_check_apis = [
            # Context类中的权限检查方法 - smali格式
            'Landroid/content/Context;->checkCallingOrSelfPermission',
            'Landroid/content/Context;->checkCallingPermission',
            'Landroid/content/Context;->checkPermission',
            'Landroid/content/Context;->checkSelfPermission',

            # PackageManager类中的权限检查方法 - smali格式
            'Landroid/content/pm/PackageManager;->checkPermission',
            'Landroid/content/pm/PackageManager;->getPermissionInfo',

            # 权限请求相关方法
            'requestPermissions',
            'onRequestPermissionsResult'
        ]

        # 定义更灵活的权限检查API模式（包含完整的smali方法签名）
        permission_check_patterns = [
            # 完整的smali方法签名模式
            r'Landroid/content/Context;->check(?:CallingOrSelf|Calling|Self)?Permission\(Ljava/lang/String;\)I',
            r'Landroid/content/Context;->enforce(?:CallingOrSelf|Calling)?Permission',
            r'Landroid/content/pm/PackageManager;->checkPermission\(Ljava/lang/String;Ljava/lang/String;\)I',
            r'Landroid/content/pm/PackageManager;->getPermissionInfo',
            r'requestPermissions\(\[Ljava/lang/String;I\)V',
            r'onRequestPermissionsResult\(I\[Ljava/lang/String;\[I\)V',

            # 支持库和兼容类
            r'Landroid/support/v4/app/ActivityCompat;->checkSelfPermission',
            r'Landroid/support/v4/content/PermissionChecker;->checkSelfPermission',
            r'Landroid/support/v4/app/ActivityCompat;->requestPermissions',
            r'Landroidx/core/app/ActivityCompat;->checkSelfPermission',
            r'Landroidx/core/content/PermissionChecker;->checkSelfPermission',
            r'Landroidx/core/app/ActivityCompat;->requestPermissions'
        ]

        # 定义权限常量模式（权限字符串通常作为参数传递）
        permission_patterns = [
            r'android\.permission\.[A-Z_]+',  # 标准权限格式
            r'com\.google\.android\.c2dm\.permission\.[A-Z_]+',  # GCM权限
            r'com\.android\.vending\.BILLING',  # 内购权限
            r'android\.permission\.C2D_MESSAGE',
            r'com\.[\w\.]+\.permission\.[A-Z_]+',  # 自定义权限
            r'android\.permission\.[\w\.]+',  # 更宽松的匹配
            r'permission\.[A-Z_]+'  # 简化的权限格式
        ]

        try:
            # 递归搜索所有smali文件
            for root_dir, _, files in os.walk(smali_dir):
                for file in files:
                    if file.endswith('.smali'):
                        file_path = os.path.join(root_dir, file)

                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                                # 方法1：检查是否包含权限检查API调用（简单匹配）
                                contains_permission_check = any(api in content for api in permission_check_apis)

                                # 方法2：使用正则表达式匹配更精确的方法签名
                                if not contains_permission_check:
                                    for pattern in permission_check_patterns:
                                        if re.search(pattern, content):
                                            contains_permission_check = True
                                            break

                                if contains_permission_check:
                                    # 提取权限字符串 - 改进的方法
                                    lines = content.split('\n')
                                    for i, line in enumerate(lines):
                                        line = line.strip()

                                        # 查找invoke指令（权限检查调用）
                                        if 'invoke-' in line:
                                            # 检查是否是权限检查API
                                            is_permission_check = False
                                            for api in permission_check_apis:
                                                if api in line:
                                                    is_permission_check = True
                                                    break

                                            # 检查正则模式
                                            if not is_permission_check:
                                                for pattern in permission_check_patterns:
                                                    if re.search(pattern, line):
                                                        is_permission_check = True
                                                        break

                                            if is_permission_check:
                                                # 向前查找可能的权限参数（const-string指令）
                                                for j in range(max(0, i-10), i):
                                                    prev_line = lines[j].strip()
                                                    # 查找const-string或const-string/jumbo指令
                                                    if 'const-string' in prev_line:
                                                        # 改进的字符串提取正则
                                                        match = re.search(r'const-string(?:/jumbo)?\s+[pv]\d+,\s*"([^"]+)"', prev_line)
                                                        if match:
                                                            string_value = match.group(1)
                                                            # 检查是否是权限字符串
                                                            for pattern in permission_patterns:
                                                                if re.search(pattern, string_value, re.IGNORECASE):
                                                                    used_permissions.add(string_value)
                                                                    logger.debug(f"在 {file_path} 中发现权限: {string_value}")
                                                                    break

                                                        # 也检查行内字符串
                                                        match = re.search(r'"([^"]+)"', prev_line)
                                                        if match:
                                                            string_value = match.group(1)
                                                            for pattern in permission_patterns:
                                                                if re.search(pattern, string_value, re.IGNORECASE):
                                                                    used_permissions.add(string_value)
                                                                    logger.debug(f"在 {file_path} 中发现权限(行内): {string_value}")
                                                                    break

                                        # 单独查找权限字符串定义
                                        if 'const-string' in line:
                                            match = re.search(r'const-string(?:/jumbo)?\s+[pv]\d+,\s*"([^"]+)"', line)
                                            if match:
                                                string_value = match.group(1)
                                                # 检查是否是权限字符串
                                                for pattern in permission_patterns:
                                                    if re.search(pattern, string_value, re.IGNORECASE):
                                                        used_permissions.add(string_value)
                                                        logger.debug(f"在 {file_path} 中发现权限字符串: {string_value}")
                                                        break

                        except Exception as e:
                            logger.warning(f"分析文件 {file_path} 失败: {e}")
                            continue

            # 使用启发式规则补充可能的权限
            if used_permissions:
                self._apply_heuristic_rules(used_permissions, smali_dir)
            else:
                logger.info("未从smali代码中发现明显的权限检查，使用声明权限作为近似")

            logger.info(f"从smali代码中分析出 {len(used_permissions)} 个实际使用的权限: {list(used_permissions)}")

        except Exception as e:
            logger.error(f"分析smali代码中的权限使用失败: {e}")

        return list(used_permissions)

    def _apply_heuristic_rules(self, used_permissions: set, smali_dir: str):
        """
        应用启发式规则补充权限

        Args:
            used_permissions: 已发现的权限集合
            smali_dir: smali代码目录路径
        """
        # 定义功能与权限的映射关系
        feature_permission_map = {
            # 网络相关
            'Landroid/net/NetworkInfo;->': 'android.permission.ACCESS_NETWORK_STATE',
            'Landroid/net/wifi/WifiManager;->': 'android.permission.ACCESS_WIFI_STATE',
            'Ljava/net/HttpURLConnection;->': 'android.permission.INTERNET',
            'Landroid/webkit/WebView;->': 'android.permission.INTERNET',
            'Lorg/apache/http/': 'android.permission.INTERNET',

            # 位置相关
            'Landroid/location/LocationManager;->': 'android.permission.ACCESS_FINE_LOCATION',
            'Landroid/location/Location;->': 'android.permission.ACCESS_FINE_LOCATION',
            'Lcom/google/android/gms/location/': 'android.permission.ACCESS_FINE_LOCATION',

            # 电话相关
            'Landroid/telephony/TelephonyManager;->': 'android.permission.READ_PHONE_STATE',
            'getDeviceId': 'android.permission.READ_PHONE_STATE',
            'getSubscriberId': 'android.permission.READ_PHONE_STATE',
            'getLine1Number': 'android.permission.READ_PHONE_STATE',

            # 短信相关
            'Landroid/telephony/SmsManager;->': 'android.permission.SEND_SMS',
            'Landroid/telephony/gsm/SmsManager;->': 'android.permission.SEND_SMS',
            'sendTextMessage': 'android.permission.SEND_SMS',

            # 存储相关
            'Ljava/io/File;->': 'android.permission.WRITE_EXTERNAL_STORAGE',
            'Landroid/os/Environment;->': 'android.permission.WRITE_EXTERNAL_STORAGE',
            'getExternalStorage': 'android.permission.WRITE_EXTERNAL_STORAGE',
            'Environment;->getExternalStorage': 'android.permission.WRITE_EXTERNAL_STORAGE',

            # 相机相关
            'Landroid/hardware/Camera;->': 'android.permission.CAMERA',
            'Landroid/media/MediaRecorder;->': 'android.permission.CAMERA',
            'Camera;->open': 'android.permission.CAMERA',

            # 录音相关
            'Landroid/media/AudioRecord;->': 'android.permission.RECORD_AUDIO',
            'MediaRecorder;->setAudioSource': 'android.permission.RECORD_AUDIO',

            # 联系人相关
            'Landroid/provider/ContactsContract;->': 'android.permission.READ_CONTACTS',
            'Landroid/content/ContentResolver;->query': 'android.permission.READ_CONTACTS',

            # 日历相关
            'Landroid/provider/CalendarContract;->': 'android.permission.READ_CALENDAR',

            # 传感器相关
            'Landroid/hardware/SensorManager;->': 'android.permission.BODY_SENSORS',

            # 账户相关
            'Landroid/accounts/AccountManager;->': 'android.permission.GET_ACCOUNTS',

            # 通知相关
            'Landroid/app/NotificationManager;->': 'android.permission.VIBRATE',
        }

        # 快速扫描smali文件中的特征API
        try:
            permission_added = False
            for root_dir, _, files in os.walk(smali_dir):
                for file in files:
                    if file.endswith('.smali'):
                        file_path = os.path.join(root_dir, file)

                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                                # 检查每个特征API
                                for api_pattern, permission in feature_permission_map.items():
                                    if api_pattern in content:
                                        # 检查权限是否已经存在
                                        if not any(permission in p for p in used_permissions):
                                            used_permissions.add(permission)
                                            permission_added = True
                                            logger.debug(f"根据API {api_pattern} 推断需要权限: {permission}")

                        except Exception as e:
                            logger.debug(f"扫描文件 {file_path} 时出错: {e}")
                            continue

            if permission_added:
                logger.info(f"通过启发式规则添加了 {len([p for p in used_permissions if any(p in fp for fp in feature_permission_map.values())])} 个推断权限")

        except Exception as e:
            logger.warning(f"应用启发式规则时出错: {e}")
    def extract_api_view(self, smali_dir: str) -> Dict:
        """
        提取API View特征
        对应论文Table 1: restricted API和suspicious API
        """
        apis = {
            'restricted_apis': [],
            'suspicious_apis': []
        }

        # 定义敏感API列表
        sensitive_apis = [
            # 系统权限相关API
            'android.telephony.TelephonyManager',
            'android.location.LocationManager',
            'android.media.MediaRecorder',
            'android.hardware.Camera',
            'android.content.ContentResolver',

            # 可疑API模式
            'getDeviceId', 'getSubscriberId', 'getLine1Number',
            'getLastKnownLocation', 'requestLocationUpdates',
            'startRecording', 'takePicture',
            'query', 'insert', 'update', 'delete'
        ]

        try:
            # 递归搜索smali文件中的API调用
            for root_dir, _, files in os.walk(smali_dir):
                for file in files:
                    if file.endswith('.smali'):
                        file_path = os.path.join(root_dir, file)
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                            # 提取API调用
                            # 匹配Landroid/...;->methodName模式
                            api_pattern = r'L([\w\/$]+);->(\w+)'
                            matches = re.findall(api_pattern, content)

                            for api_class, api_method in matches:
                                api_full_name = f"{api_class.replace('/', '.')}.{api_method}"

                                # 判断是否为敏感API
                                is_sensitive = any(pattern in api_full_name for pattern in sensitive_apis)
                                is_restricted = any(keyword in api_full_name for keyword in ['telephony', 'location', 'camera', 'content'])

                                if is_restricted:
                                    apis['restricted_apis'].append(api_full_name)
                                elif is_sensitive:
                                    apis['suspicious_apis'].append(api_full_name)

            logger.info(f"提取到 {len(apis['restricted_apis'])} 个受限API和 {len(apis['suspicious_apis'])} 个可疑API")

        except Exception as e:
            logger.error(f"提取API特征失败: {e}")

        return apis

    def extract_url_uses_feature_view(self, decompiled_files: Dict) -> Dict:
        """
        提取URL & uses-feature View特征
        对应论文Table 1: URL和uses-feature
        """
        features = {
            'urls': [],
            'uses_features': []
        }

        try:
            # 提取URL特征 (从代码和资源中)
            # 搜索smali代码中的URL
            if 'smali_dir' in decompiled_files:
                for root_dir, _, files in os.walk(decompiled_files['smali_dir']):
                    for file in files:
                        if file.endswith('.smali'):
                            file_path = os.path.join(root_dir, file)
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                                # 匹配URL模式
                                url_pattern = r'https?://[^\s<>"\']+|\bwww\.[^\s<>"\']+'
                                urls = re.findall(url_pattern, content)
                                features['urls'].extend(urls)

            # 提取uses-feature特征
            manifest_file = decompiled_files.get('manifest')
            if manifest_file and os.path.exists(manifest_file):
                tree = ET.parse(manifest_file)
                root = tree.getroot()

                for uses_feature in root.findall('.//uses-feature'):
                    feature_name = uses_feature.get('{http://schemas.android.com/apk/res/android}name')
                    if feature_name:
                        features['uses_features'].append(feature_name)

            # 去重
            features['urls'] = list(set(features['urls']))
            features['uses_features'] = list(set(features['uses_features']))

            logger.info(f"提取到 {len(features['urls'])} 个URL和 {len(features['uses_features'])} 个uses-feature")

        except Exception as e:
            logger.error(f"提取URL和uses-feature特征失败: {e}")

        return features

    def extract_all_features(self, seq: int) -> Dict:
        """
            提取所有特征 (三视图特征)
            返回结构化的特征字典
        """
        features = {
            'seq': seq,
            'permission_view': {},
            'api_view': {},
            'url_uses_feature_view': {}
        }

        # 获取APK路径
        apk_path = self.get_apk_path_by_seq(seq)
        if not apk_path:
            return features

        # 创建临时反编译目录
        temp_dir = f"temp_decompile_{seq}"

        # 反编译APK
        decompiled_files = self.decompile_apk(apk_path, temp_dir)
        if not decompiled_files:
            return features

        try:
            # 修改这里：传递smali_dir参数给extract_permission_view
            features['permission_view'] = self.extract_permission_view(
                decompiled_files['manifest'],
                decompiled_files['smali_dir']  # 新增参数
            )

            # 提取API View特征
            features['api_view'] = self.extract_api_view(
                decompiled_files['smali_dir']
            )

            # 提取URL & uses-feature View特征
            features['url_uses_feature_view'] = self.extract_url_uses_feature_view(
                decompiled_files
            )

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

            logger.info(f"成功提取序列 {seq} 的所有特征")

        except Exception as e:
            logger.error(f"提取特征过程中出错: {e}")
            # 确保清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        return features
    def batch_extract_features(self, train_file: str, output_file: str = 'extracted_features.json'):
        """
        批量提取特征

        Args:
            train_file: 训练集文件路径
            output_file: 特征输出文件
        """
        # 加载序列号
        seqs = self.load_seqs_from_file(train_file)
        if not seqs:
            logger.error("未找到有效的序列号")
            return

        all_features = {}

        for i, seq in enumerate(seqs):
            logger.info(f"正在提取序列 {seq} 的特征 ({i+1}/{len(seqs)})")

            features = self.extract_all_features(seq)
            all_features[seq] = features

            # 每处理10个序列保存一次
            if (i + 1) % 10 == 0:
                self._save_features_intermediate(all_features, output_file)

        # 保存最终结果
        self._save_features_intermediate(all_features, output_file)
        logger.info(f"特征提取完成，结果保存到 {output_file}")

    def _save_features_intermediate(self, features: Dict, output_file: str):
        """保存中间特征结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存特征文件失败: {e}")

def main(TRAIN_FILE = "train_0.8repartition_little.txt",
    OUTPUT_FILE = "apppoet_extracted_features.json"):
    """主函数"""
    # 配置参数

    # 创建特征提取器
    extractor = AppPoetFeatureExtractor()

    # 批量提取特征
    extractor.batch_extract_features(TRAIN_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    TRAIN_FILE = "train_0.8repartition_little.txt"
    OUTPUT_FILE = "apppoet_extracted_features.json"
    main(TRAIN_FILE,OUTPUT_FILE)