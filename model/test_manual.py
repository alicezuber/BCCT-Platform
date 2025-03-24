import requests
import json
import sys

def test_api():
    print("===== 測試四則運算 API =====")
    
    base_url = "http://localhost:5000/calculate"
    print(f"目標 API 地址: {base_url}")
    print("開始測試...\n")
    
    try:
        # Test addition
        print("測試加法:")
        response = requests.post(
            base_url,
            json={"operation": "+", "num1": 10, "num2": 5}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: {'result': 15}")
        print("測試結果: " + ("通過" if response.json().get('result') == 15 else "失敗"))
        print("-" * 40)
        
        # Test subtraction
        print("測試減法:")
        response = requests.post(
            base_url,
            json={"operation": "-", "num1": 10, "num2": 5}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: {'result': 5}")
        print("測試結果: " + ("通過" if response.json().get('result') == 5 else "失敗"))
        print("-" * 40)
        
        # Test multiplication
        print("測試乘法:")
        response = requests.post(
            base_url,
            json={"operation": "*", "num1": 10, "num2": 5}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: {'result': 50}")
        print("測試結果: " + ("通過" if response.json().get('result') == 50 else "失敗"))
        print("-" * 40)
        
        # Test division
        print("測試除法:")
        response = requests.post(
            base_url,
            json={"operation": "/", "num1": 10, "num2": 5}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: {'result': 2.0}")
        print("測試結果: " + ("通過" if response.json().get('result') == 2.0 else "失敗"))
        print("-" * 40)
        
        # Test division by zero
        print("測試除以零:")
        response = requests.post(
            base_url,
            json={"operation": "/", "num1": 10, "num2": 0}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: 狀態碼 400 和錯誤訊息")
        print("測試結果: " + ("通過" if response.status_code == 400 and 'error' in response.json() else "失敗"))
        print("-" * 40)
        
        # Test invalid operation
        print("測試無效運算:")
        response = requests.post(
            base_url,
            json={"operation": "^", "num1": 10, "num2": 5}
        )
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {response.json()}")
        print("預期結果: 狀態碼 400 和錯誤訊息")
        print("測試結果: " + ("通過" if response.status_code == 400 and 'error' in response.json() else "失敗"))
        print("-" * 40)
        
        print("\n===== 測試摘要 =====")
        print("所有測試已完成")
        
    except requests.exceptions.ConnectionError:
        print("錯誤: 無法連接到 API 伺服器。請確保 Flask 伺服器正在運行。")
        print("提示: 在另一個終端運行 'python api.py' 來啟動伺服器。")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤: 測試過程中發生未預期的錯誤: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
