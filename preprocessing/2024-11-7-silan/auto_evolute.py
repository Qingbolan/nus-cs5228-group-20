import sys
import os
import importlib

# 保存当前工作目录
original_working_dir = os.getcwd()

# 获取项目根目录路径并添加到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# 临时切换到项目根目录
os.chdir(project_root)

# 导入rmse_calculator
from data.rmse_calculator import calculate_rmse

# 使用importlib导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

preprocess_path = os.path.join(current_dir, 'preprocess.py')
lgm_release1_path = os.path.join(current_dir, 'tempCodeRunnerFile.py')

preprocess_module = import_module_from_path('preprocess', preprocess_path)
lgm_release1_module = import_module_from_path('lgm_release1', lgm_release1_path)

preprocess_main = preprocess_module.main
lgm_release1_main = lgm_release1_module.main

# 切换回原始工作目录
os.chdir(original_working_dir)

# 现在可以使用导入的模块了
# your code here...

from dotenv import load_dotenv
import os

load_dotenv()

print('Preprocessing data...')
preprocess_main()
print('Preprocessing data done.')
print('Training model...')
lgm_release1_main()
print('Training model done.')
# print('Calculating RMSE...')
# RMSE = calculate_rmse(os.getenv('output_prediction'))
# print(f"RMSE Score: {RMSE:.4f}")
# print('All done.')

# if RMSE>= 21429.3845:
#     print('RSME not good enough')
# else:
#     print('RSME good enough')