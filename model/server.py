from flask import Flask, request, jsonify
from flask_cors import CORS  # 新增 CORS 匯入
import os
from dotenv import load_dotenv  # 導入 dotenv 支援

# 載入環境變數
load_dotenv()

app = Flask(__name__)
CORS(app)  # 啟用 CORS 支援

@app.route('/api/add', methods=['GET', 'POST'])
def add():
    try:
        # 處理GET請求
        if request.method == 'GET':
            a = float(request.args.get('a', 0))
            b = float(request.args.get('b', 0))
        # 處理POST請求
        else:
            data = request.get_json()
            a = float(data.get('a', 0))
            b = float(data.get('b', 0))
        
        result = a + b
        return jsonify({
            'success': True,
            'result': result,
            'operation': f"{a} + {b}",
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '請提供有效的數字參數'
        }), 400

if __name__ == '__main__':
    # 從環境變數中讀取 IP 地址
    host_ip = os.getenv('ShiLaCow_ip', '127.0.0.1')  # 若找不到則預設為 127.0.0.1
    app.run(debug=True, host=host_ip, port=5000)
