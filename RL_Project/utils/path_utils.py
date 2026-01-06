"""
路径工具 - 确保正确的模块导入路径
"""
import os
import sys

def setup_path():
    """设置项目根目录到Python路径"""
    # 获取项目根目录（RL_Project）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 添加到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

# 自动设置路径
setup_path()

