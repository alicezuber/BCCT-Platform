# 加法 API 使用說明

這是一個簡單的加法API，可以計算兩個數字的和並返回結果。

## 安裝與啟動

### 1. 安裝需要的套件

```bash
pip install flask flask-cors python-dotenv
```

### 2. 啟動服務器

```bash
python server.py
```

服務器將運行在 http://25.7.171.251:5000 (由 .env 文件中的 ShiLaCow_ip 設定)

## API 使用方式

### 1. GET 請求

發送GET請求到 `/api/add` 端點，並通過URL參數傳遞兩個數字。

#### 請求格式
```
GET /api/add?a=數字1&b=數字2
```

#### 範例
```
GET http://25.7.171.251:5000/api/add?a=5&b=3
```

#### 使用curl
```bash
curl "http://25.7.171.251:5000/api/add?a=5&b=3"
```

### 2. POST 請求

發送POST請求到 `/api/add` 端點，並通過JSON格式傳遞兩個數字。

#### 請求格式
```
POST /api/add
Content-Type: application/json

{
  "a": 數字1,
  "b": 數字2
}
```

#### 範例
```bash
curl -X POST http://25.7.171.251:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

## 回應格式

### 成功回應

```json
{
  "success": true,
  "result": 8,
  "operation": "5.0 + 3.0"
}
```

### 失敗回應

```json
{
  "success": false,
  "error": "錯誤訊息",
  "message": "請提供有效的數字參數"
}
```

## 錯誤處理

- 如果提供的參數不是有效的數字，API將返回400錯誤。
- 確保在請求中提供必要的參數 `a` 和 `b`。

## 環境配置

- API 服務器 IP 地址由 `.env` 文件中的 `ShiLaCow_ip` 變數設定。
- 如需修改服務器 IP，請編輯 `.env` 文件而非直接修改程式碼。
