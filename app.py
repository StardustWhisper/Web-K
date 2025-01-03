from flask import Flask, render_template, send_from_directory, request, jsonify, send_file
import os
from datetime import datetime, timedelta
import chart
from config import CHARTS_DIR

app = Flask(__name__)

def get_chart_info(filename):
    """解析图表文件名，获取图表信息"""
    # 移除.png后缀
    name = filename.replace('.png', '')
    parts = name.split('_')
    
    # 解析时间戳
    timestamp = None
    for part in parts[-2:]:  # 检查最后两个部分
        try:
            timestamp = datetime.strptime(part, '%Y%m%d')
            break
        except ValueError:
            try:
                timestamp = datetime.strptime(f"{parts[-2]}_{parts[-1]}", '%Y%m%d_%H%M%S')
                break
            except ValueError:
                continue
    
    # 获取文件的实际修改时间作为备选
    if not timestamp:
        file_path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(file_path):
            timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
    
    # 提取股票代码和名称
    code = parts[0] if len(parts) > 0 else ''
    name = parts[1] if len(parts) > 1 else ''
    
    # 如果是上证指数，特殊处理
    if code == '000001' and not name:
        name = '上证指数'
    
    info = {
        'filename': filename,
        'code': code,
        'name': name,
        'type': '对数坐标' if '对数' in filename else '普通坐标',
        'days': next((p.replace('天', '') for p in parts if '天' in p), '全部'),
        'timestamp': timestamp.strftime('%Y-%m-%d') if timestamp else '未知时间',
        'path': f'charts/{filename}'
    }
    return info

@app.route('/')
def index():
    # 获取charts目录下的所有图表
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith('.png')]
    chart_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHARTS_DIR, x)), reverse=True)
    
    # 解析每个图表的信息
    charts = [get_chart_info(f) for f in chart_files]
    
    return render_template('index.html', charts=charts)

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory(CHARTS_DIR, filename)

@app.route('/generate', methods=['POST'])
def generate_chart():
    """处理生成图表的请求"""
    try:
        # 获取表单数据
        code = request.form.get('code', '')
        days = request.form.get('days', '')
        log_scale = request.form.get('log_scale', '')
        adjust = request.form.get('adjust', 'qfq')
        
        # 转换参数类型
        days = int(days) if days.isdigit() else None
        
        # 生成图表
        plt = chart.plot_stock_candlestick(
            stock_code=code if code else None,
            days=days,
            log_scale=log_scale if log_scale != 'none' else None,
            adjust=adjust
        )
        
        if plt is not None:
            plt.close()
            return jsonify({'status': 'success', 'message': '图表生成成功'})
        else:
            return jsonify({'status': 'error', 'message': '图表生成失败'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download/<path:filename>')
def download_chart(filename):
    """下载图表"""
    try:
        return send_file(
            os.path.join(CHARTS_DIR, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete/<path:filename>', methods=['POST'])
def delete_chart(filename):
    """删除图表"""
    try:
        file_path = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'status': 'success', 'message': '删除成功'})
        else:
            return jsonify({'status': 'error', 'message': '文件不存在'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stock_info', methods=['GET'])
def get_stock_info():
    """获取股票信息"""
    try:
        code = request.args.get('code', '')
        if not code:
            # 默认返回上证指数信息
            return jsonify({
                'status': 'success',
                'data': {
                    'name': '上证指数',
                    'code': '000001',
                    'market': 'sh'
                }
            })
            
        # 使用akshare获取股票信息
        import akshare as ak
        
        # 标准化股票代码格式
        code = str(code).zfill(6)  # 补齐6位
        
        try:
            # 先尝试获取股票日K数据，验证股票代码是否有效
            test_df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            
            if test_df.empty:
                return jsonify({
                    'status': 'error',
                    'message': '无效的股票代码'
                })
            
            # 获取股票名称
            stock_info = ak.stock_individual_info_em(symbol=code)
            
            if not stock_info.empty:
                # 从stock_info中提取股票名称
                name_row = stock_info[stock_info['item'] == '股票简称']
                stock_name = name_row['value'].iloc[0] if not name_row.empty else f'股票{code}'
            else:
                stock_name = f'股票{code}'
            
            return jsonify({
                'status': 'success',
                'data': {
                    'name': stock_name,
                    'code': code,
                    'market': 'sh' if code.startswith('6') else 'sz'
                }
            })
            
        except Exception as e:
            print(f"获取股票信息时出错: {e}")
            # 如果发生错误但是股票代码格式正确，返回基本信息
            if (code.startswith(('0', '3', '6')) and 
                len(code) == 6 and 
                code.isdigit()):
                return jsonify({
                    'status': 'success',
                    'data': {
                        'name': f'股票{code}',
                        'code': code,
                        'market': 'sh' if code.startswith('6') else 'sz'
                    }
                })
            return jsonify({
                'status': 'error',
                'message': '无效的股票代码'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/stock_data', methods=['GET'])
def get_stock_data():
    """获取股票历史数据"""
    try:
        code = request.args.get('code', '')
        days = request.args.get('days', '')
        adjust = request.args.get('adjust', 'qfq')
        
        import akshare as ak
        
        if not days or not days.isdigit():
            # 如果没有指定天数，返回全部历史数据
            start_date = "19900101"
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            # 使用二分查找来确定正确的起始日期
            target_days = int(days)
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 初始尝试：使用2倍的自然日作为查找范围
            initial_days = target_days * 2
            
            def get_data(start_date, end_date):
                if not code:  # 获取上证指数数据
                    return ak.index_zh_a_hist(
                        symbol="000001",
                        period="daily",
                        start_date=start_date,
                        end_date=end_date
                    )
                else:  # 获取股票数据
                    return ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
            
            # 二分查找合适的起始日期
            left_days = target_days
            right_days = initial_days
            final_df = None
            
            while left_days <= right_days:
                mid_days = (left_days + right_days) // 2
                start_date = (datetime.now() - timedelta(days=mid_days)).strftime('%Y%m%d')
                
                df = get_data(start_date, end_date)
                current_days = len(df)
                
                if current_days == target_days:
                    final_df = df
                    break
                elif current_days < target_days:
                    # 如果获取到的交易日数量小于目标天数，增加查找范围
                    left_days = mid_days + 1
                else:
                    # 如果获取到的交易日数量大于目标天数，缩小范围并保存结果
                    final_df = df.tail(target_days)
                    right_days = mid_days - 1
            
            if final_df is not None:
                df = final_df
            else:
                # 如果二分查找失败，使用最后一次查询的结果并截取
                start_date = (datetime.now() - timedelta(days=initial_days)).strftime('%Y%m%d')
                df = get_data(start_date, end_date)
                df = df.tail(target_days)
        
        # 转换数据格式
        data = df.to_dict('records')
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)