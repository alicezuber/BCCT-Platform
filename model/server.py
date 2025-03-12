from flask import Flask, request, jsonify

app = Flask(__name__)

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
    app.run(debug=True, host='0.0.0.0', port=5000)
