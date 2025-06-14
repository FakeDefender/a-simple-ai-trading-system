import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.utils.config_loader import load_config
from src.utils.data_loader import DataLoader

if __name__ == "__main__":
    config = load_config()
    data_loader = DataLoader(config)
    print("开始测试 get_benchmark_data ...")
    benchmark_data = data_loader.get_benchmark_data()
    if benchmark_data is not None and not benchmark_data.empty:
        print("获取成功！数据预览：")
        print(benchmark_data.head())
        print("数据基本信息：")
        print(benchmark_data.info())
        print("缺失值统计：")
        print(benchmark_data.isnull().sum())
    else:
        print("获取失败或数据为空。") 