# 這是關於檔案的一些資訊及規則

## model - 模型


## my-new-dapp - 平台
只需要了解下面幾項就好;

- 區塊鏈
  - build(這個類似函式庫，裡面記載合約裡的內容。)
  本地需要有才能執行有關區塊鏈的程式，**不然會錯誤說找不到;**

  - contracts(存放合約)
  如果不是開發成員別理他;

  - migrations(合約上鏈程式)
  如果不是開發成員別理他;

  - test(測試區塊鏈)
  如果不是開發成員別理他;

  - truffle-config.js(設定檔)

- 平台(前端)
  - pages(前端頁面)
    - api(網頁從伺服器連接到其他外部服務)
    - styles(放css)

- 連接或共用
  - lib(存放連接庫)
    - db.js(資料庫連結)
    - 剩下的(區塊鏈合約翻譯)

  - .env.local(環境檔)
  目前存放連接資料庫的位址跟加密的密鑰

  - .gitignore(存放沒有要上傳到github的檔名資料)

  - README.md(使用教學)


## rule - 規則
 - about_file.md
 介紹檔案內容

 - about_version.md
 介紹github版本上傳規格

## version - 版本
各自的版本歷史紀錄

## 其他
 - TODO.MD
 代辦事項

 - .env 
 整個專案的環境變數(目前只有ip)
 由於.env不能上傳 所以有需要分享的請使用.env.example

 - .gitignore
 忽略不上傳gh的檔案

 - README.md
 專案介紹