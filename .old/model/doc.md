# Git 安裝教學

## 簡介

Git 是一個分散式版本控制系統，用於追蹤檔案的變化，並協調多人共同開發專案的工作。在開始使用 Git 之前，你必須先在你的電腦上安裝它。本文檔將指導你如何在不同的作業系統上安裝 Git。

## 版本要求

本文檔基於 Git 2.0.0 版本。雖然較舊版本的 Git 通常也能支援大部分指令，但某些功能可能會有差異或無法使用。Git 提供了良好的向下相容性，所以如果你使用的 Git 版本高於 2.0，應該不會有太大問題。

## 在不同平台上安裝 Git

### Linux 系統

在 Linux 上安裝 Git 最簡單的方法是使用發行版的套件管理系統。

#### Fedora 或 RHEL 系列 (CentOS, Scientific Linux)

```bash
$ sudo yum install git-all
```

#### Debian 系列 (Ubuntu)

```bash
$ sudo apt-get install git-all
```

如需其他 Linux 發行版的安裝指南，請參考 [Git 官方網站](https://git-scm.com/download/linux)。

### macOS 系統

在 macOS 上安裝 Git 有多種方法：

1. **使用 Xcode 命令列工具**：
   - 在 macOS Mavericks (10.9) 或更新版本，只需在終端機中輸入 `git` 命令，如果尚未安裝，系統會提示你安裝。

2. **使用官方安裝程式**：
   - 從 [Git 官方網站](https://git-scm.com/download/mac) 下載最新版本的安裝程式。

3. **使用 GitHub Desktop**：
   - 安裝 [GitHub Desktop](http://mac.github.com)，它包含了 Git 命令列工具。

### Windows 系統

在 Windows 上安裝 Git 也有多種方法：

1. **使用官方安裝程式**：
   - 從 [Git 官方網站](https://git-scm.com/download/win) 下載安裝程式。
   - 這是一個名為 Git for Windows 的專案，與 Git 本身是獨立的。
   - 更多資訊請參考 [Git for Windows](http://git-for-windows.github.io/)。

2. **使用 GitHub Desktop**：
   - 安裝 [GitHub Desktop](http://windows.github.com)，它包含了 Git 命令列工具和圖形化介面。
   - 它能完美搭配 PowerShell，設定實體憑證快取和 CRLF 設定。

## 從原始碼安裝

如果你想要安裝最新版本的 Git，可以考慮從原始碼安裝。這種方法通常能獲得比二進位安裝程式更新的版本。

### 安裝相依套件

首先，你需要安裝 Git 所需的函式庫：

#### Fedora 或 RHEL 系列

```bash
$ sudo yum install curl-devel expat-devel gettext-devel \
  openssl-devel perl-devel zlib-devel
```

#### Debian 系列

```bash
$ sudo apt-get install libcurl4-gnutls-dev libexpat1-dev gettext \
  libz-dev libssl-dev
```

### 安裝文件格式相依套件

如果你需要建立文件 (doc、html、info)，還需要安裝這些額外的套件：

#### Fedora 或 RHEL 系列

```bash
$ sudo yum install asciidoc xmlto docbook2X
$ sudo ln -s /usr/bin/db2x_docbook2texi /usr/bin/docbook2x-texi
```

#### Debian 系列

```bash
$ sudo apt-get install asciidoc xmlto docbook2x
```

### 下載並編譯原始碼

你可以從以下網站下載最新的 Git 原始碼：
- [Kernel.org](https://www.kernel.org/pub/software/scm/git)
- [GitHub](https://github.com/git/git/releases)

下載後，編譯並安裝 Git：

```bash
$ tar -zxf git-2.0.0.tar.gz
$ cd git-2.0.0
$ make configure
$ ./configure --prefix=/usr
$ make all doc info
$ sudo make install install-doc install-html install-info
```

## 安裝後的驗證

安裝完成後，你可以在終端機或命令提示字元中輸入以下命令來確認 Git 已正確安裝：

```bash
$ git --version
```

這應該會顯示你安裝的 Git 版本號碼。

## 後續步驟

安裝完 Git 後，建議進行初次設定，包括設定你的使用者名稱和電子郵件地址：

```bash
$ git config --global user.name "你的名字"
$ git config --global user.email "你的電子郵件"
```

## 資源

- [Git 官方網站](https://git-scm.com/)
- [Git 官方文件](https://git-scm.com/doc)
- [Pro Git 書籍](https://git-scm.com/book/zh-tw/v2)

---

本文檔參考自 [Git 官方文件](https://git-scm.com/book/zh-tw/v2/%E9%96%8B%E5%A7%8B-Git-%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8)。
