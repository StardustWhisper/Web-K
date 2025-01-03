import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import traceback
import argparse
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from threading import Lock
import threading
import logging
from functools import lru_cache
import time
import matplotlib
matplotlib.use('Agg')  # For systems without display
from matplotlib.font_manager import FontProperties
import os
from config import CHARTS_DIR, FONTS_DIR

# Configure thread-safe logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_lock = Lock()
# Global locks
plt_lock = threading.Lock()
data_lock = threading.Lock()
log_lock = threading.Lock()

def log_safe(message, level=logging.INFO):
    with log_lock:
        logger.log(level, message)

def get_stock_info(stock_code):
    """获取股票信息"""
    try:
        if not stock_code or stock_code == '000001':
            return {
                'code': '000001',
                'name': '上证指数',
                'market': 'sh',
                'type': 'index'
            }
            
        # 标准化股票代码格式
        stock_code = str(stock_code).zfill(6)
        
        # 先验证股票代码格式
        if not (stock_code.isdigit() and len(stock_code) == 6):
            return None
            
        try:
            # 获取实时行情数据（包含市值信息）
            realtime_info = ak.stock_zh_a_spot_em()
            stock_realtime = realtime_info[realtime_info['代码'] == stock_code]
            
            # 获取股票基本信息
            stock_info = ak.stock_individual_info_em(symbol=stock_code)
            
            # 提取所需信息
            info_dict = {
                'code': stock_code,
                'type': 'stock',
                'market': 'sh' if stock_code.startswith('6') else 'sz'
            }
            
            # 从stock_info中提取基本信息
            if not stock_info.empty:
                info_mapping = {
                    '股票简称': 'name',
                    '行业': 'industry',
                    '上市时间': 'listing_date'
                }
                
                for item, key in info_mapping.items():
                    row = stock_info[stock_info['item'] == item]
                    if not row.empty:
                        info_dict[key] = row['value'].iloc[0]
            
            # 添加市值信息（从实时行情中获取）
            if not stock_realtime.empty:
                # 将市值从亿元转换为数值
                total_mv = float(stock_realtime['总市值'].iloc[0])
                float_mv = float(stock_realtime['流通市值'].iloc[0])
                
                info_dict.update({
                    'total_market_value': total_mv,
                    'float_market_value': float_mv,
                    'pe_ratio': stock_realtime['市盈率-动态'].iloc[0],
                    'pb_ratio': stock_realtime['市净率'].iloc[0]
                })
            
            # 获取财务指标
            try:
                # 获取最新财务指标
                fin_df = ak.stock_financial_analysis_indicator(symbol=stock_code)
                if not fin_df.empty:
                    latest = fin_df.iloc[0]
                    # 添加财务指标，使用准确的列名
                    fin_indicators = {
                        'roe': latest.get('净资产收益率(%)', 'N/A'),  # 保持不变
                        'roa': latest.get('总资产利润率(%)', 'N/A'),  # 更改为更准确的指标
                        'profit_margin': latest.get('销售净利率(%)', 'N/A'),  # 保持不变
                        'debt_ratio': latest.get('资产负债率(%)', 'N/A'),  # 保持不变
                        'eps': latest.get('摊薄每股收益(元)', 'N/A'),  # 更改为摊薄每股收益
                        'bps': latest.get('每股净资产_调整后(元)', 'N/A'),  # 更改为调整后的每股净资产
                        'revenue_yoy': latest.get('主营业务收入增长率(%)', 'N/A'),  # 更改为更准确的指标
                        'profit_yoy': latest.get('净利润增长率(%)', 'N/A'),  # 保持不变
                        
                        # 添加更多有价值的财务指标
                        'gross_profit_margin': latest.get('销售毛利率(%)', 'N/A'),  # 新增：毛利率
                        'operating_margin': latest.get('营业利润率(%)', 'N/A'),  # 保持不变
                        'current_ratio': latest.get('流动比率', 'N/A'),  # 保持不变
                        'quick_ratio': latest.get('速动比率', 'N/A'),  # 保持不变
                        
                        # 新增一些重要的运营指标
                        'asset_turnover': latest.get('总资产周转率(次)', 'N/A'),  # 新增：总资产周转率
                        'inventory_turnover_days': latest.get('存货周转天数(天)', 'N/A'),  # 新增：存货周转天数
                        'receivables_turnover_days': latest.get('应收账款周转天数(天)', 'N/A'),  # 新增：应收账款周转天数
                        'equity_ratio': latest.get('股东权益比率(%)', 'N/A'),  # 新增：股东权益比率
                    }
                    
                    info_dict.update(fin_indicators)
            except Exception as e:
                print(f"获取财务指标时出错: {e}")
            
            # 确保name字段存在
            if 'name' not in info_dict or not info_dict['name']:
                info_dict['name'] = f'股票{stock_code}'
            
            return info_dict
            
        except Exception as e:
            print(f"获取股票信息时出错: {e}")
            if stock_code.startswith(('0', '3', '6')):
                return {
                    'code': stock_code,
                    'name': f'股票{stock_code}',
                    'market': 'sh' if stock_code.startswith('6') else 'sz',
                    'type': 'stock'
                }
            return None
            
    except Exception as e:
        print(f"获取股票信息失败: {e}")
        return None

def calculate_macd(df, short=12, long=26, signal=9):
    """
    计算MACD指标
    :param df: 包含收盘价的DataFrame
    :param short: 短期EMA周期
    :param long: 长期EMA周期
    :param signal: 信号线周期
    :return: 包含MACD指标的DataFrame
    """
    # 计算短期和长期EMA
    exp1 = df['close'].ewm(span=short, adjust=False).mean()
    exp2 = df['close'].ewm(span=long, adjust=False).mean()
    
    # 计算DIF
    df['DIF'] = exp1 - exp2
    # 计算DEA (MACD Signal)
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    # 计算MACD柱状图
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    return df

def add_golden_lines(ax, df, log_scale=None):
    """
    添加黄金分割线
    参数:
        ax: matplotlib轴对象
        df: 数据框
        log_scale: 对数坐标类型，可选值：'10'-常用对数, 'e'-自然对数, None-普通坐标
    """
    import numpy as np
    
    # 获取最高价和最低价
    high_price = df['high'].max()
    low_price = df['low'].min()
    
    # 黄金分割比例
    fib_levels = [0, 0.146, 0.236, 0.382, 0.5, 0.618, 0.786, 0.854, 1]
    
    # 根据坐标系类型计算分割线位置
    if log_scale:
        # 在对数空间中计算
        log_func = np.log10 if log_scale == '10' else np.log
        log_min = log_func(low_price)
        log_max = log_func(high_price)
        log_range = log_max - log_min
        
        prices = [np.power(10, log_min + level * log_range) if log_scale == '10' 
                 else np.exp(log_min + level * log_range) for level in fib_levels]
    else:
        # 线性空间中计算
        price_range = high_price - low_price
        prices = [low_price + price_range * level for level in fib_levels]
    
    # 计算标签位置（统一在右侧）
    x_pos = len(df) * 1.045  # 靠右显示
    
    # 绘制分割线和标签
    for price, level in zip(prices, fib_levels):
        # 绘制水平线
        ax.axhline(y=price, color='gray', linestyle='--', alpha=0.3)
        
        # 计算相对于最低点的涨跌幅
        change_percent = (price - low_price) / low_price * 100
        
        # 生成标签文本
        if level in [0, 1]:  # 最高点和最低点显示特殊标签
            label = f'{"最高" if level == 1 else "最低"}点: {price:.2f}'
        else:
            label = f'{level*100:.1f}% ({price:.2f})'
        
        # 绘制标签
        ax.text(x_pos, price, label,
                va='center',      # 垂直居中对齐
                ha='left',       # 左对齐
                fontsize=8,      # 字体大小
                bbox=dict(
                    facecolor='white',    # 白色背景
                    edgecolor='none',     # 无边框
                    alpha=0.7,            # 透明度
                    pad=1                 # 内边距
                ))

def find_best_text_position(ax, df):
    """
    找到文本框的最佳位置，强制放置在边缘位置
    """
    try:
        # 获取数据范围
        y_max = df['high'].max()
        y_min = df['low'].min()
        price_range = y_max - y_min
        
        # 计算左右两侧的K线位置
        left_margin = int(len(df) * 0.1)  # 增加左侧边距到10%
        right_margin = int(len(df) * 0.9)  # 减少右侧边距到90%
        
        # 获取最左侧和最右侧的数据
        left_data = df.iloc[:left_margin]
        right_data = df.iloc[right_margin:]
        
        # 计算左右两侧的最高价和最低价
        left_max = left_data['high'].max()
        left_min = left_data['low'].min()
        right_max = right_data['high'].max()
        right_min = right_data['low'].min()
        
        # 强制选择右侧位置
        x = - len(df) * 0.02  # 靠近最右边
        
        # 计算合适的y轴位置
        # 选择价格区间最高点，给予足够的上方空间
        y_position = y_max - price_range * 0.03  # 距离顶部3%的位置
        
        return {
            'x': x,
            'y': y_position,
            'halign': 'left',
            'valign': 'top'
        }
        
    except Exception as e:
        print(f"计算文本位置时出错: {e}")
        # 返回一个安全的默认位置
        return {
            'x': len(df) * 0.02,
            'y': df['high'].max(),
            'halign': 'left',
            'valign': 'top'
        }

def calculate_technical_indicators(df):
    """
    计算技术指标：MA、MACD等
    """
    try:
        # Make sure required columns exist
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Calculate indicators
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        
        # Calculate MACD
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        
        return df
    except Exception as e:
        log_safe(f"Error calculating technical indicators: {e}", logging.ERROR)
        return df

def setup_log_scale(ax, log_base='10'):
    """
    设置对数坐标系
    参数:
        ax: matplotlib轴对象
        log_base: 对数的底数，'10'或'e'
    """
    import numpy as np
    
    if log_base == '10':
        ax.set_yscale('log', base=10)
        # 设置y轴格式化函数，显示实际价格
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y))
        )
    elif log_base == 'e':
        ax.set_yscale('log', base=np.e)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y))
        )
    
    # 优化刻度显示
    ax.yaxis.set_minor_formatter(
        plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y))
    )
    
    # 设置网格线
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

def setup_chinese_font():
    """设置中文字体"""
    # 设置备选字体列表
    font_list = [
        'Noto Sans CJK JP',
        'Noto Sans CJK SC',
        'DejaVu Sans',
        'Liberation Sans',
        'Arial',
        'Helvetica'
    ]
    
    # 配置matplotlib字体
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 尝试使用系统字体
    for font_name in font_list:
        try:
            font = FontProperties(family=font_name)
            # 测试字体是否可用
            if font.get_name() != 'DejaVu Sans':  # 默认回退字体
                return font
        except:
            continue
    
    # 如果没有找到合适的字体，使用默认字体
    return FontProperties()

def generate_chart_filename(code, name, days=None, log_scale=None, adjust='qfq'):
    """生成图表文件名"""
    # 获取当前时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 构建文件名部分
    parts = []
    
    # 添加代码和名称（确保都存在）
    code = str(code) if code else '000001'
    name = str(name) if name else '上证指数'
    parts.append(f"{code}_{name}")
    
    # 添加其他信息
    if days:
        parts.append(f"{days}天")
    if log_scale:
        parts.append(f"对数坐标{log_scale}")
    if adjust != 'qfq':
        parts.append('后复权' if adjust == 'hfq' else '不复权')
    
    # 添加时间戳
    parts.append(current_time)
    
    # 组合文件名
    filename = '_'.join(parts) + '.png'
    
    # 清理文件名中的非法字符（保留中文字符）
    filename = ''.join(c for c in filename if c.isalnum() or c in '_-.' or '\u4e00' <= c <= '\u9fff')
    
    return filename

def format_volume_axis(value, p):
    """
    格式化成交量显示，使用万手/亿手为单位
    """
    if value >= 1_0000_0000:  # 1亿手以上
        return f'{value/1_0000_0000:.1f}亿手'
    elif value >= 10000:  # 1万手以上
        return f'{value/10000:.1f}万手'
    else:
        return f'{int(value):,}手'

def format_amount_axis(value, p):
    """
    格式化成交额显示，使用万元/亿元为单位
    """
    if value >= 1_0000_0000:  # 1亿以上
        return f'{value/1_0000_0000:.1f}亿'
    elif value >= 10000:  # 1万以上
        return f'{value/10000:.1f}万'
    else:
        return f'{int(value):,}'

def format_price_axis(value, p):
    """
    格式化价格显示，如果是整数则不显示小数点
    """
    if value == int(value):  # 如果是整数
        return f'{int(value)}'
    else:
        return f'{value:.2f}'  # 保留两位小数

def format_index_price(value, p):
    """指数专用的格式化函数"""
    # 将数值转换为字符串，检查小数部分
    str_value = f"{value:.2f}"
    int_part, dec_part = str_value.split('.')
    
    # 如果小数部分全是0，则返回整数
    if dec_part == '00':
        return int_part
    # 如果小数第二位是0，则只保留一位小数
    elif dec_part.endswith('0'):
        return f"{value:.1f}"
    # 其他情况保留两位小数
    else:
        return str_value

def plot_stock_candlestick(stock_code=None, index=None, days=None, log_scale=None, adjust='qfq'):
    """绘制股票K线图"""
    with plt_lock:
        try:
            # 设置中文字体
            font = setup_chinese_font()
            
            # 创建图表
            fig = plt.figure(figsize=(16, 12))  # 调整整体大小
            
            # 创建网格布局，调整各个子图的高度比例
            gs = GridSpec(3, 1, height_ratios=[6, 2, 2], hspace=0.1)
            
            # K线图
            ax1 = plt.subplot(gs[0])
            # MACD
            ax2 = plt.subplot(gs[1])
            # 成交量和成交额
            ax3 = plt.subplot(gs[2])
            ax3_twin = ax3.twinx()
            
            # 获取股票信息
            if index == 'sse' or (not stock_code and not index):
                stock_info = {
                    'code': '000001',
                    'name': '上证指数',
                    'market': 'sh',
                    'type': 'index'
                }
                is_index = True
            else:
                stock_info = get_stock_info(stock_code)
                if not stock_info:
                    raise ValueError(f"无法获取股票信息: {stock_code}")
                is_index = stock_info.get('type') == 'index'
            
            # 获取数据
            df = get_stock_data(stock_code if not is_index else None, days, adjust)
            if df is None or df.empty:
                raise ValueError("无法获取股票数据")
            
            # 重命名列并确保数据类型正确
            if is_index:
                column_mappings = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
            else:
                column_mappings = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
            df = df.rename(columns=column_mappings)

            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # 确保数值列为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 获取最新价格和涨跌幅
            if len(df) >= 2:  # 确保至少有两行数据
                latest_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2]
                price_change = ((latest_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
            else:
                latest_price = df['close'].iloc[-1] if len(df) > 0 else 0
                price_change = 0

            # 计算技术指标
            df = calculate_technical_indicators(df)

            # 绘制K线图
            width = 0.6
            width2 = width * 0.8
            
            # 计算每个时间点的x坐标
            x = np.arange(len(df.index))
            
            # 上涨和下跌日期
            up = df['close'] >= df['open']
            down = df['close'] < df['open']
            
            # 绘制K线实体
            ax1.bar(x[up], df['close'][up] - df['open'][up], width2,
                   bottom=df['open'][up], color='red', alpha=0.7)
            ax1.bar(x[down], df['close'][down] - df['open'][down], width2,
                   bottom=df['open'][down], color='green', alpha=0.7)
            
            # 绘制上下影线
            ax1.vlines(x[up], df['low'][up], df['high'][up],
                      color='red', linewidth=1, alpha=0.7)
            ax1.vlines(x[down], df['low'][down], df['high'][down],
                      color='green', linewidth=1, alpha=0.7)
            
            # 设置对数坐标（如果需要）
            if log_scale:
                setup_log_scale(ax1, log_scale)
            
            # 添加均线
            ax1.plot(x, df['MA5'], 'b--', alpha=0.8, label='MA5')
            ax1.plot(x, df['MA10'], 'y--', alpha=0.8, label='MA10')
            ax1.plot(x, df['MA20'], 'g--', alpha=0.8, label='MA20')
            ax1.plot(x, df['MA60'], 'r--', alpha=0.8, label='MA60')
            
            # 添加K线图图例
            legend_elements = [
                Line2D([0], [0], color='b', linestyle='--', alpha=0.8, label='MA5'),
                Line2D([0], [0], color='y', linestyle='--', alpha=0.8, label='MA10'),
                Line2D([0], [0], color='g', linestyle='--', alpha=0.8, label='MA20'),
                Line2D([0], [0], color='r', linestyle='--', alpha=0.8, label='MA60'),
                Line2D([0], [0], color='none', label='|'),  # 分隔符
                Patch(facecolor='red', alpha=0.7, label='上涨'),
                Patch(facecolor='green', alpha=0.7, label='下跌'),
                Line2D([0], [0], color='none', label='|'),  # 分隔符
                Line2D([0], [0], color='none', label=f'{get_adjust_name(adjust)}'),
                Line2D([0], [0], color='none', label=f'{get_scale_name(log_scale)}'),
                Line2D([0], [0], color='none', label=f'交易日: {len(df)}')
            ]
            
            # 添加图例到K线图，设置为单排水平排列
            ax1.legend(handles=legend_elements,
                      loc='upper left',
                      bbox_to_anchor=(0.01, 0.06),
                      fontsize=6,
                      framealpha=0.2,
                      prop=font,
                      ncol=11,  # 所有元素在一行
                      columnspacing=0.5,  # 减小列间距
                      handlelength=1.0,  # 减小图例线条长度
                      handletextpad=0.3)  # 减小文本和图例符号的间距
            
            # 添加黄金分割线
            add_golden_lines(ax1, df, log_scale)
            
            # 绘制MACD
            ax2.plot(x, df['DIF'], 'blue', label='DIF')
            ax2.plot(x, df['DEA'], 'orange', label='DEA')
            ax2.bar(x, df['MACD'], color=['red' if m >= 0 else 'green' for m in df['MACD']], 
                   alpha=0.7, width=0.8)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            
            # 绘制成交量和成交额
            # 成交量柱状图
            volume_colors = ['red' if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                           else 'green' for i in range(len(df))]
            volume_line = ax3_twin.bar(x, df['volume'],  # 使用 x 而不是 df.index
                                      color=volume_colors, 
                                      alpha=0.5,
                                      width=0.6)
            
            # 添加成交额折线图
            amount_line = ax3.plot(x, df['amount'],  # 使用 x 而不是 df.index
                                  color='blue', 
                                  alpha=0.8,
                                  linewidth=1,
                                  label='成交额')
            
            # 设置成交额和成交量的y轴格式
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_amount_axis))
            ax3_twin.yaxis.set_major_formatter(plt.FuncFormatter(format_volume_axis))
            
            # 设置y轴标签颜色，与对应的线条颜色匹配
            ax3.yaxis.label.set_color('blue')
            ax3_twin.yaxis.label.set_color('black')
            ax3.tick_params(axis='y', colors='blue')
            ax3_twin.tick_params(axis='y', colors='black')
            
            # 设置标题和图例
            # ax3.set_title('成交量/成交额', fontproperties=font)
            
            # 合并两个y轴的图例
            lines = volume_line
            labels = ['成交量']
            # 创建组合图例
            legend_elements = [
                Line2D([0], [0], color='blue', label='成交额'),
                Patch(facecolor='red', alpha=0.5, label='成交量(涨)'),
                Patch(facecolor='green', alpha=0.5, label='成交量(跌)')
            ]
            ax3.legend(handles=legend_elements, 
                      loc='upper left', 
                      prop=font, 
                      framealpha=0.8)
            
            # 设置所有子图的x轴格式
            def format_date(x, p):
                if x >= 0 and x < len(df.index):
                    return df.index[int(x)].strftime('%Y-%m-%d')
                return ''
            
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
                plt.setp(ax.get_xticklabels(), 
                        rotation=0,
                        ha='center',
                        fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if len(df) > 30:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            
            # 设置标题
            if is_index:
                title = f'上证指数(000001) K线图\n{df.index[0].strftime("%Y-%m-%d")} 至 {df.index[-1].strftime("%Y-%m-%d")}'
            else:
                title = f'{stock_info["name"]}({stock_code}) K线图\n{df.index[0].strftime("%Y-%m-%d")} 至 {df.index[-1].strftime("%Y-%m-%d")}'
            
            # 计算价格范围
            price_range = df['high'].max() - df['low'].min()
            y_margin = price_range * 0.05  # 设置10%的边距
            
            # 设置y轴范围
            y_min = df['low'].min() - y_margin
            y_max = df['high'].max() + y_margin
            
            # 设置K线图的显示范围
            ax1.set_ylim(y_min, y_max)
            
            # 设置x轴范围，留出合适的边距
            x_margin = len(df) * 0.04  # 设置5%的边距
            ax1.set_xlim(-x_margin, len(df) - 1 + x_margin)
            
            # 调整文本框位置
            text_pos = find_best_text_position(ax1, df)
            info_text = format_info_text(stock_info, latest_price, price_change)
            
            # 添加文本框，调整字体大小和透明度
            ax1.text(text_pos['x'], text_pos['y'],
                    info_text,
                    transform=ax1.transData,
                    bbox=dict(facecolor='white', 
                             alpha=0.6,
                             edgecolor='none',
                             boxstyle='round,pad=0.5'),
                    verticalalignment=text_pos['valign'],
                    horizontalalignment=text_pos['halign'],
                    fontsize=8,
                    fontproperties=font)
            
            # 设置主标题
            ax1.set_title(title, fontproperties=font, pad=10, loc='center')
            
            # 在子图内部添加标题
            # MACD标题
            ax2.text(0.02, 0.95, 'MACD', 
                    transform=ax2.transAxes,
                    fontproperties=font,
                    fontsize=8,
                    verticalalignment='top')
            
            # 调整布局，确保所有元素都能完整显示
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 保存图表
            try:
                # 生成文件名
                filename = generate_chart_filename(
                    stock_info['code'],
                    stock_info['name'],
                    days,
                    log_scale,
                    adjust
                )
                filepath = os.path.join(CHARTS_DIR, filename)
                
                plt.savefig(filepath, 
                           dpi=300, 
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none')
                print(f"图表已保存为: {filepath}")
            except Exception as e:
                print(f"保存图表时出错: {e}")
                backup_path = os.path.join(CHARTS_DIR, 'chart_backup.png')
                plt.savefig(backup_path, dpi=300, bbox_inches='tight')
                print(f"已保存备份图表: {backup_path}")
            
            # 添加MACD图例
            ax2.legend(loc='upper left', prop=font, framealpha=0.8)
            
            return plt

        except Exception as e:
            log_safe(f"Error: {str(e)}", logging.ERROR)
            traceback.print_exc()
            return None

def format_market_value(value):
    """
    格式化市值显示
    输入值单位为元，输出按照万/亿/万亿显示
    """
    try:
        # 确保输入是数值类型
        value = float(value)
        
        # 转换单位（1万=10000，1亿=100000000，1万亿=1000000000000）
        if value >= 1000000000000:  # 大于等于1万亿
            return f"{value/1000000000000:.2f}万亿"
        elif value >= 100000000:  # 大于等于1亿
            return f"{value/100000000:.2f}亿"
        elif value >= 10000:  # 大于等于1万
            return f"{value/10000:.2f}万"
        else:  # 小于1万
            return f"{value:.2f}元"
    except (ValueError, TypeError):
        return str(value)

def format_info_text(stock_info, latest_price, price_change):
    """格式化股票信息文本"""
    try:
        # 基本信息
        text = [
            f"名称：{stock_info.get('name', 'N/A')}",
            f"代码：{stock_info.get('code', 'N/A')}",
            f"最新价格：{latest_price:.2f}",
            f"涨跌幅：{price_change:+.2f}%"
        ]
        
        # 区分股票和指数的信息显示
        if stock_info.get('type') == 'index':
            # [指数信息显示部分保持不变...]
            pass
        else:
            # 股票信息显示
            # 1. 市场信息
            if stock_info.get('industry'):
                text.append(f"所属行业：{stock_info['industry']}")
            if stock_info.get('total_market_value'):
                text.append(f"总市值：{format_market_value(stock_info['total_market_value'])}")
            if stock_info.get('float_market_value'):
                text.append(f"流通市值：{format_market_value(stock_info['float_market_value'])}")
            
            # 2. 估值指标
            if stock_info.get('pe_ratio') not in ['', 'N/A', '-']:
                text.append(f"市盈率(动)：{stock_info['pe_ratio']}")
            if stock_info.get('pb_ratio') not in ['', 'N/A', '-']:
                text.append(f"市净率：{stock_info['pb_ratio']}")
            
            # 3. 财务指标
            text.append("\n财务指标：")
            
            # 收益能力
            if stock_info.get('eps') not in ['', 'N/A', '-']:
                text.append(f"摊薄每股收益：{stock_info['eps']}元")
            if stock_info.get('bps') not in ['', 'N/A', '-']:
                text.append(f"每股净资产：{stock_info['bps']}元")
            if stock_info.get('roe') not in ['', 'N/A', '-']:
                text.append(f"ROE：{stock_info['roe']}%")
            if stock_info.get('roa') not in ['', 'N/A', '-']:
                text.append(f"ROA：{stock_info['roa']}%")
            
            # 盈利能力
            if stock_info.get('gross_profit_margin') not in ['', 'N/A', '-']:
                text.append(f"毛利率：{stock_info['gross_profit_margin']}%")
            if stock_info.get('operating_margin') not in ['', 'N/A', '-']:
                text.append(f"营业利润率：{stock_info['operating_margin']}%")
            if stock_info.get('profit_margin') not in ['', 'N/A', '-']:
                text.append(f"净利率：{stock_info['profit_margin']}%")
            
            # 偿债能力
            if stock_info.get('current_ratio') not in ['', 'N/A', '-']:
                text.append(f"流动比率：{stock_info['current_ratio']}")
            if stock_info.get('quick_ratio') not in ['', 'N/A', '-']:
                text.append(f"速动比率：{stock_info['quick_ratio']}")
            if stock_info.get('debt_ratio') not in ['', 'N/A', '-']:
                text.append(f"资产负债率：{stock_info['debt_ratio']}%")
            
            # 上市时间
            if stock_info.get('listing_date'):
                text.append(f"\n上市时间：{stock_info['listing_date']}")
        
        return '\n'.join(text)
    except Exception as e:
        print(f"格式化信息文本时出错: {e}")
        return f"名称：{stock_info.get('name', 'N/A')}\n代码：{stock_info.get('code', 'N/A')}"

def get_adjust_name(adjust):
    """获取复权类型的中文名称"""
    adjust_names = {
        'qfq': '前复权',
        'hfq': '后复权',
        'none': '不复权'
    }
    return adjust_names.get(adjust, adjust)

def get_scale_name(log_scale):
    """获取坐标类型的中文名称"""
    if not log_scale:
        return '普通坐标'
    elif log_scale == '10':
        return '对数坐标(10)'
    elif log_scale == 'e':
        return '对数坐标(e)'
    return f'对数坐标({log_scale})'

# Add caching for stock info with timeout
@lru_cache(maxsize=128)
def get_stock_info_cached(stock_code):
    return get_stock_info(stock_code)

# Add cache invalidation
def invalidate_cache():
    get_stock_info_cached.cache_clear()

def get_stock_data(stock_code=None, days=None, adjust='qfq'):
    """获取股票数据"""
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        
        if stock_code:  # 获取股票数据
            # 使用二分查找获取准确的交易日数量
            if days:
                left_days = days
                right_days = days * 2  # 初始范围
                target_days = days
                final_df = None
                
                while left_days <= right_days:
                    mid_days = (left_days + right_days) // 2
                    start_date = (datetime.now() - timedelta(days=mid_days)).strftime('%Y%m%d')
                    
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
                    
                    current_days = len(df)
                    if current_days == target_days:
                        final_df = df
                        break
                    elif current_days < target_days:
                        left_days = mid_days + 1
                    else:
                        final_df = df.tail(target_days)
                        right_days = mid_days - 1
                
                if final_df is not None:
                    df = final_df
                else:
                    # 如果二分查找失败，使用最后一次查询的结果
                    start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
                    df = df.tail(target_days)
            else:
                # 如果没有指定天数，获取全部历史数据
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date="19900101",
                    end_date=end_date,
                    adjust=adjust
                )
        else:  # 获取上证指数数据
            # 对指数数据使用相同的逻辑
            if days:
                left_days = days
                right_days = days * 2
                target_days = days
                final_df = None
                
                while left_days <= right_days:
                    mid_days = (left_days + right_days) // 2
                    start_date = (datetime.now() - timedelta(days=mid_days)).strftime('%Y%m%d')
                    
                    df = ak.index_zh_a_hist(
                        symbol="000001",
                        period="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    current_days = len(df)
                    if current_days == target_days:
                        final_df = df
                        break
                    elif current_days < target_days:
                        left_days = mid_days + 1
                    else:
                        final_df = df.tail(target_days)
                        right_days = mid_days - 1
                
                if final_df is not None:
                    df = final_df
                else:
                    start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')
                    df = ak.index_zh_a_hist(
                        symbol="000001",
                        period="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
                    df = df.tail(target_days)
            else:
                df = ak.index_zh_a_hist(
                    symbol="000001",
                    period="daily",
                    start_date="19900101",
                    end_date=end_date
                )
        
        return df
        
    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        traceback.print_exc()
        return None

def main():
    """
    主函数
    """
    try:
        # Remove font cache clearing
        # Invalidate cache every hour
        if time.time() % 3600 < 60:  # Check if within first minute of the hour
            invalidate_cache()
            
        # 获取命令行参数
        parser = argparse.ArgumentParser(description='股票K线图绘制工具')
        parser.add_argument('--code', type=str, help='股票代码, 例如: 600000')
        parser.add_argument('--index', type=str, choices=['sse', 'sz'],
                          help='指数类型: sse-上证指数, sz-深圳指数')
        parser.add_argument('--days', type=int, help='显示天数,不填则显示全部数据', default=None)
        parser.add_argument('--log', type=str, nargs='?', const='10', choices=['10', 'e'],
                          help='使用对数坐标: 10-常用对数, e-自然对数')
        parser.add_argument('--adjust', type=str, choices=['qfq', 'hfq', 'none'],
                          default='qfq', help='复权类型: qfq-前复权, hfg-后复权, none-不复权')
        args = parser.parse_args()
        
        # 处理参数
        stock_code = args.code if args.code else None
        days = args.days
        log_scale = args.log
        adjust = args.adjust
        
        # 打印参数信息
        print(f"\n股票代码: {stock_code if stock_code else '上证指数'}")
        print(f"显示天数: {days if days else '全部'}")
        print(f"坐标类型: {'对数(底数=' + log_scale + ')' if log_scale else '普通'}")
        print(f"复权类型: {'前复权' if adjust == 'qfq' else '后复权' if adjust == 'hfq' else '不复权'}")
        
        # 调用绘图函数
        plt = plot_stock_candlestick(stock_code, args.index, days, log_scale, adjust)
        if plt is not None:
            # 保存图表后关闭
            plt.close()
            print("图表已生成完成")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Remove the clear_font_cache() call
    main()
