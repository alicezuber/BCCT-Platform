# BCCT-platform
不管你愉不愉快，總之合作愉快

# 專案簡介
BCCT-Platform 是一個區塊鏈技術整合平台，包含前端網頁應用、API 服務和相關模型。本專案旨在提供區塊鏈技術的實際應用案例與開發環境。

# 注意

TODO list 請努力完成，也可以向其他部分提出要求

# 環境配置
更詳細請看/rule/about_file.md


API 服務器 IP 地址由 .env 文件中的 ShiLaCow_ip 變數設定
編譯 truffle 時使用 Node.js v16
其他開發環境使用 Node.js v18
建議複製 .env.example 為 .env 並根據個人環境進行配置

# 安裝與使用
前端應用

```
cd my-new-dapp
npm install
npm run dev
```

瀏覽器訪問: http://localhost:3000

API 服務

```
cd model
pip install flask flask-cors python-dotenv
python server.py
````

API 服務將運行在 .env 檔案中的 ShiLaCow_ip 配置的地址上。


# 專案結構
更詳細請看/rule/about_file.md
```
├── model/              # API 服務與模型
│   ├── README.md       # API 使用說明
│   └── server.py       # Flask 服務器
├── my-new-dapp/        # Next.js 前端應用
├── version/            # 各成員版本記錄
└── README.md           # 專案說明
```

# Learning Resources

- 如何使用 Github
https://hackmd.io/@sysprog/git-with-github

- 為什麼要用 Next.js

https://ithelp.ithome.com.tw/articles/10265138

- 版本控制

https://www.getguru.com/zh/reference/document-version-control


# 關於此專案
在編譯truffle時 版本使用node v16 除此之外用v18

### 版本管理
更詳細請看/rule/about_version.md

本專案使用獨立的版本文件進行版本管理，格式為：[vX.Y.Z][使用者代號] 更新內容

- 主版本號 (X): 重大功能變更或不向下兼容的變更
- 次版本號 (Y): 新增功能但保持向下兼容
- 修補版本號 (Z): 錯誤修復和小調整

各成員版本記錄請參考 version 資料夾中的對應文件。

# Contributors

[![Minatobaiyun](https://avatars.githubusercontent.com/u/92900313?v=4)](https://github.com/alicezuber)

使用者代號: B

[![LALAboom](https://avatars.githubusercontent.com/u/106247967?v=4)](https://github.com/IWannaListenToUSayUwU)

使用者代號: S

[![abcd1070070](https://avatars.githubusercontent.com/u/103110383?v=4)](https://github.com/abcd1070070)

使用者代號: C

[![abcd1070070](https://avatars.githubusercontent.com/u/50486110?v=4)](https://github.com/tcy129)

使用者代號: T

主分支

使用者代號: M