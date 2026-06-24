import argparse
import platform
import sys

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AppPoet")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_2")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_final")

from AppPoet import apppoet_multi_view_prompt, apppoet_detection_classifier, apppoet_feature_extractor
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='训练集文件路径')
    parser.add_argument('--test', required=True, help='测试集文件路径')
    args = parser.parse_args()

    train_file = args.train
    test_file = args.test
    train_feature_file = "features_train.json"
    test_feature_file = "features_test.json"
    train_llm_output_file = "train_llm_output_file.json"
    test_llm_output_file = "test_llm_output_file.json"
    model_path = "apppoet_detection_model.pth"
    #训练集 抽取特征，llm特征增强
    apppoet_feature_extractor.main(train_file, train_feature_file)
    apppoet_multi_view_prompt.main(train_feature_file, train_file, train_llm_output_file)
    #测试集 抽取特征，llm特征增强
    apppoet_feature_extractor.main(test_file, test_feature_file)
    apppoet_multi_view_prompt.main(test_feature_file, test_file, test_llm_output_file)
    #模型训练和检测
    apppoet_detection_classifier.main(train_llm_output_file, model_path, test_llm_output_file, test_file)
