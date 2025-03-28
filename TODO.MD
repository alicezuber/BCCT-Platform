# TODO List

## 大範圍
- [ ] 範例 - by.M
- [ ] 請將IP設置讀取.env，以利大家自部屬 - by.B

## 模型
- [ ] 範例 - by.M

## 工具
- [ ] 範例 - by.M

## 網頁
- [ ] 範例 - by.M

## 模型API
- [ ] 範例 - by.M

-----------------------
## ShiLaCow
- [login] -在api/login有個await connectToDatabase();
導致每次重開伺服器時 登入時才連到資料庫 使得連線很久
所以需要改寫-開伺服器時連到資料庫 不要要登入時在連
