import os

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义各种路径
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# 创建必要的目录
for directory in [CHARTS_DIR, FONTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory) 