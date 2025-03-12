import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import re
import os
import time
import json

class GitHelperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Git 操作輔助工具")
        self.root.geometry("800x600")
        
        # 獲取當前工作目錄
        self.repo_path = os.path.dirname(os.path.abspath(__file__))
        
        # 版本文件路徑
        self.version_file_path = os.path.join(self.repo_path, "version.md")
        
        # README文件路徑
        self.readme_path = os.path.join(self.repo_path, "README.md")
        
        # 用戶設定檔路徑
        self.config_path = os.path.join(self.repo_path, ".git_helper_config.json")
        
        # 設置主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 版本訊息
        ttk.Label(self.main_frame, text="BCCT-platform Git 輔助工具 v1.0.0", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 顯示當前分支
        self.branch_frame = ttk.LabelFrame(self.main_frame, text="當前分支")
        self.branch_frame.pack(fill=tk.X, pady=5)
        
        self.current_branch_var = tk.StringVar()
        self.current_branch_label = ttk.Label(self.branch_frame, textvariable=self.current_branch_var, font=("Arial", 12))
        self.current_branch_label.pack(pady=5)
        
        # 分支選擇區
        self.branch_select_frame = ttk.LabelFrame(self.main_frame, text="切換分支")
        self.branch_select_frame.pack(fill=tk.X, pady=5)
        
        self.branches_var = tk.StringVar()
        self.branch_combobox = ttk.Combobox(self.branch_select_frame, textvariable=self.branches_var, state="readonly", width=50)
        self.branch_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.switch_btn = ttk.Button(self.branch_select_frame, text="切換", command=self.switch_branch)
        self.switch_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.refresh_btn = ttk.Button(self.branch_select_frame, text="刷新分支列表", command=self.refresh_branches)
        self.refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Commit區域
        self.commit_frame = ttk.LabelFrame(self.main_frame, text="提交更改")
        self.commit_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 版本號輸入
        self.version_frame = ttk.Frame(self.commit_frame)
        self.version_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.version_frame, text="版本號:").pack(side=tk.LEFT, padx=5)
        
        self.major_var = tk.StringVar(value="0")
        self.minor_var = tk.StringVar(value="0")
        self.patch_var = tk.StringVar(value="0")
        
        ttk.Entry(self.version_frame, textvariable=self.major_var, width=3).pack(side=tk.LEFT)
        ttk.Label(self.version_frame, text=".").pack(side=tk.LEFT)
        ttk.Entry(self.version_frame, textvariable=self.minor_var, width=3).pack(side=tk.LEFT)
        ttk.Label(self.version_frame, text=".").pack(side=tk.LEFT)
        ttk.Entry(self.version_frame, textvariable=self.patch_var, width=3).pack(side=tk.LEFT)
        
        # 添加自動獲取版本按鈕
        self.get_version_btn = ttk.Button(self.version_frame, text="從version.md獲取版本", command=self.get_version_from_file)
        self.get_version_btn.pack(side=tk.LEFT, padx=10)
        
        # 在版本號輸入區添加使用者代號選擇
        self.user_code_frame = ttk.Frame(self.version_frame)
        self.user_code_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(self.user_code_frame, text="使用者代號:").pack(side=tk.LEFT)
        
        # 使用者代號下拉選單
        self.user_code_var = tk.StringVar()
        self.user_code_combobox = ttk.Combobox(self.user_code_frame, textvariable=self.user_code_var, state="readonly", width=5)
        self.user_code_combobox.pack(side=tk.LEFT, padx=5)
        
        # 提交訊息輸入
        ttk.Label(self.commit_frame, text="提交訊息:").pack(anchor=tk.W, padx=5)
        
        self.commit_message = scrolledtext.ScrolledText(self.commit_frame, height=5)
        self.commit_message.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 預覽提交訊息
        ttk.Label(self.commit_frame, text="預覽:").pack(anchor=tk.W, padx=5)
        
        self.preview_var = tk.StringVar()
        ttk.Entry(self.commit_frame, textvariable=self.preview_var, state="readonly", width=80).pack(fill=tk.X, padx=5, pady=5)
        
        # 按鈕區域
        self.button_frame = ttk.Frame(self.commit_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        self.preview_btn = ttk.Button(self.button_frame, text="生成預覽", command=self.generate_preview)
        self.preview_btn.pack(side=tk.LEFT, padx=5)
        
        self.commit_btn = ttk.Button(self.button_frame, text="提交(Commit)", command=self.commit_changes)
        self.commit_btn.pack(side=tk.LEFT, padx=5)
        
        self.push_btn = ttk.Button(self.button_frame, text="推送(Push)", command=self.push_changes)
        self.push_btn.pack(side=tk.LEFT, padx=5)
        
        self.pull_btn = ttk.Button(self.button_frame, text="拉取(Pull)", command=self.pull_changes)
        self.pull_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_btn = ttk.Button(self.button_frame, text="檢視狀態", command=self.show_status)
        self.status_btn.pack(side=tk.LEFT, padx=5)
        
        self.merge_btn = ttk.Button(self.button_frame, text="合併至main", command=self.show_merge_dialog)
        self.merge_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加檢查同步狀態的按鈕
        self.check_sync_btn = ttk.Button(self.button_frame, text="檢查遠端差異", command=self.check_sync_status)
        self.check_sync_btn.pack(side=tk.LEFT, padx=5)
        
        # 日誌區域 - 確保先創建日誌區域，然後才調用需要日誌功能的方法
        self.log_frame = ttk.LabelFrame(self.main_frame, text="操作日誌")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_area = scrolledtext.ScrolledText(self.log_frame, height=8)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_area.config(state=tk.DISABLED)
        
        # 在日誌區域初始化後再讀取用戶代號
        self.user_codes = self.read_user_codes()
        if self.user_codes:
            self.user_code_combobox["values"] = self.user_codes
            # 嘗試從配置讀取上次使用的代號
            last_code = self.load_config().get("last_user_code")
            if last_code and last_code in self.user_codes:
                self.user_code_var.set(last_code)
        
        # 初始化
        self.refresh_branches()
        self.update_current_branch()
        
        # 嘗試在啟動時自動獲取版本號
        self.get_version_from_file()
        
        # 綁定事件
        self.major_var.trace("w", lambda *args: self.generate_preview())
        self.minor_var.trace("w", lambda *args: self.generate_preview())
        self.patch_var.trace("w", lambda *args: self.generate_preview())
        self.user_code_var.trace("w", lambda *args: self.generate_preview())
        self.commit_message.bind("<KeyRelease>", lambda e: self.generate_preview())
    
    def log(self, message):
        """向日誌區域添加消息"""
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)
    
    def run_git_command(self, command, show_output=True):
        """運行Git命令並返回結果"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            if show_output and result.stdout:
                self.log(f"命令輸出:\n{result.stdout}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            error_msg = f"命令失敗: {e.stderr}"
            self.log(error_msg)
            if show_output:
                messagebox.showerror("錯誤", error_msg)
            return None
    
    def update_current_branch(self):
        """更新顯示當前分支"""
        branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        if branch:
            self.current_branch_var.set(f"當前分支: {branch}")
        else:
            self.current_branch_var.set("無法獲取當前分支")
    
    def refresh_branches(self):
        """刷新分支列表"""
        branches_output = self.run_git_command(["git", "branch"], show_output=False)
        if branches_output:
            branches = []
            for line in branches_output.split("\n"):
                branch = line.strip()
                if branch.startswith("*"):
                    branch = branch[1:].strip()  # 移除當前分支標記，修正拼寫錯誤trip->strip
                branches.append(branch)
            
            self.branch_combobox["values"] = branches
            # 選擇當前分支
            current = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
            if current in branches:
                self.branches_var.set(current)
            
            self.log(f"分支列表已更新，共 {len(branches)} 個分支")
            self.update_current_branch()
    
    def switch_branch(self):
        """切換到選定分支"""
        branch = self.branches_var.get()
        if not branch:
            messagebox.showwarning("警告", "請選擇要切換的分支")
            return
        
        # 檢查有無未提交的變更
        status = self.run_git_command(["git", "status", "--porcelain"], show_output=False)
        if status:
            confirm = messagebox.askyesno(
                "未提交變更",
                f"有未提交的變更可能會丟失:\n{status}\n\n確定要切換分支嗎?"
            )
            if not confirm:
                return
        
        result = self.run_git_command(["git", "checkout", branch])
        if result is not None:
            self.log(f"已切換到分支: {branch}")
            self.update_current_branch()
    
    def generate_preview(self):
        """生成提交訊息預覽"""
        major = self.major_var.get()
        minor = self.minor_var.get()
        patch = self.patch_var.get()
        user_code = self.user_code_var.get()
        message = self.commit_message.get("1.0", tk.END).strip()
        
        # 版本號驗證
        try:
            int(major)
            int(minor)
            int(patch)
        except ValueError:
            self.preview_var.set("[版本號無效] " + message)
            return
        
        version = f"v{major}.{minor}.{patch}"
        
        # 添加使用者代號
        if user_code:
            preview = f"[{version}][{user_code}] {message}"
            # 保存最後使用的代號
            config = self.load_config()
            config["last_user_code"] = user_code
            self.save_config(config)
        else:
            preview = f"[{version}] {message}"
        
        self.preview_var.set(preview)
    
    def commit_changes(self):
        """提交更改"""
        commit_msg = self.preview_var.get()
        if not commit_msg:
            messagebox.showwarning("警告", "提交訊息不能為空")
            return
        
        # 檢查提交訊息格式是否正確
        # 支援兩種格式：[vX.Y.Z] 訊息 或 [vX.Y.Z][代號] 訊息
        if not re.match(r'^\[v\d+\.\d+\.\d+\](\[[A-Za-z0-9]\])? .+$', commit_msg):
            messagebox.showwarning("警告", "提交訊息格式不正確，應為 [vX.Y.Z] 訊息 或 [vX.Y.Z][代號] 訊息")
            return
        
        # 檢查是否有文件要提交
        status = self.run_git_command(["git", "status", "--porcelain"], show_output=False)
        if not status:
            messagebox.showinfo("訊息", "沒有文件需要提交")
            return
        
        # 添加所有文件
        self.run_git_command(["git", "add", "."])
        
        # 提交
        result = self.run_git_command(["git", "commit", "-m", commit_msg])
        if result is not None:
            messagebox.showinfo("成功", "變更已提交")
    
    def push_changes(self):
        """推送變更到遠程倉庫"""
        current_branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        if not current_branch:
            messagebox.showwarning("警告", "無法獲取當前分支")
            return
        
        confirm = messagebox.askyesno("確認", f"要將變更推送到 {current_branch} 分支嗎?")
        if not confirm:
            return
        
        self.log(f"正在推送到 {current_branch}...")
        result = self.run_git_command(["git", "push", "origin", current_branch])
        if result is not None:
            messagebox.showinfo("成功", f"變更已推送到 {current_branch}")
    
    def pull_changes(self):
        """從遠程倉庫拉取變更"""
        current_branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        if not current_branch:
            messagebox.showwarning("警告", "無法獲取當前分支")
            return
        
        confirm = messagebox.askyesno("確認", f"要從遠程倉庫拉取 {current_branch} 分支的變更嗎?")
        if not confirm:
            return
        
        self.log(f"正在從 origin/{current_branch} 拉取...")
        result = self.run_git_command(["git", "pull", "origin", current_branch])
        if result is not None:
            messagebox.showinfo("成功", "變更已拉取")
            # 刷新顯示
            self.update_current_branch()
    
    def show_status(self):
        """顯示當前倉庫狀態"""
        status = self.run_git_command(["git", "status"])
        self.log("倉庫狀態:\n" + status if status else "獲取狀態失敗")
    
    def get_version_from_file(self):
        """從version.md文件中獲取最新版本號和提交訊息"""
        if not os.path.exists(self.version_file_path):
            self.log("找不到version.md文件")
            messagebox.showwarning("警告", f"找不到version.md文件: {self.version_file_path}")
            return
            
        try:
            with open(self.version_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # 使用正則表達式匹配版本行，例如 ## 1.0.6
            version_pattern = r'##\s+(\d+)\.(\d+)\.(\d+)'
            matches = re.findall(version_pattern, content)
                
            if not matches:
                self.log("未在version.md中找到版本號")
                messagebox.showwarning("警告", "未在version.md中找到版本號")
                return
                
            # 取第一個匹配（最新版本）
            major, minor, patch = matches[0]
            
            # 設置界面輸入框 - 版本號
            self.major_var.set(major)
            self.minor_var.set(minor)
            self.patch_var.set(patch)
            
            # 提取最新版本的提交信息
            latest_version = f"## {major}.{minor}.{patch}"
            version_index = content.find(latest_version)
            
            if (version_index != -1):
                # 找出該版本的提交信息區塊
                next_version_pattern = r'##\s+\d+\.\d+\.\d+'
                next_match = re.search(next_version_pattern, content[version_index + len(latest_version):])
                
                if next_match:
                    version_content = content[version_index + len(latest_version):version_index + len(latest_version) + next_match.start()]
                else:
                    version_content = content[version_index + len(latest_version):]
                
                # 提取所有以 "- " 開頭的行
                commit_items = re.findall(r'- (.*?)(?:\n|$)', version_content)
                
                if commit_items:
                    # 合併所有提交項目為一個訊息
                    commit_message = "; ".join(commit_items)
                    # 清除可能存在的舊訊息並設置新訊息
                    self.commit_message.delete('1.0', tk.END)
                    self.commit_message.insert('1.0', commit_message)
                    self.log(f"已從version.md獲取提交訊息")
            
            self.log(f"已從version.md獲取最新版本: {major}.{minor}.{patch}")
                
            # 更新預覽
            self.generate_preview()
                
        except Exception as e:
            error_msg = f"讀取version.md時發生錯誤: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("錯誤", error_msg)
    
    def show_merge_dialog(self):
        """顯示合併至main的對話框"""
        # 檢查是否位於main分支
        current_branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        if current_branch == "main":
            messagebox.showwarning("警告", "您當前已在main分支，無需合併")
            return
        
        # 創建對話框
        merge_dialog = tk.Toplevel(self.root)
        merge_dialog.title("合併至main")
        merge_dialog.geometry("400x300")
        merge_dialog.transient(self.root)
        merge_dialog.grab_set()  # 模態對話框
        
        # 讀取上一個版本號
        last_version = self.get_last_version()
        if not last_version:
            messagebox.showerror("錯誤", "無法讀取上一個版本號")
            merge_dialog.destroy()
            return
        
        major, minor, patch = last_version
        
        # 顯示當前版本
        ttk.Label(merge_dialog, text="當前版本:").pack(pady=10, padx=10, anchor=tk.W)
        ttk.Label(merge_dialog, text=f"v{major}.{minor}.{patch}", font=("Arial", 12, "bold")).pack(pady=5, padx=10)
        
        # 選擇版本號增加方式
        ttk.Label(merge_dialog, text="選擇版本增加方式:").pack(pady=10, padx=10, anchor=tk.W)
        
        version_choice = tk.StringVar(value="patch")
        
        # 使用單選按鈕讓使用者選擇
        # 大版本 - 主要功能變動
        major_choice = ttk.Radiobutton(
            merge_dialog, 
            text=f"主版本號 (v{int(major)+1}.0.0) - 重大功能變更", 
            variable=version_choice, 
            value="major"
        )
        major_choice.pack(pady=5, padx=20, anchor=tk.W)
        
        # 中版本 - 小功能新增
        minor_choice = ttk.Radiobutton(
            merge_dialog, 
            text=f"次版本號 (v{major}.{int(minor)+1}.0) - 新功能添加", 
            variable=version_choice, 
            value="minor"
        )
        minor_choice.pack(pady=5, padx=20, anchor=tk.W)
        
        # 小版本 - 修復bug
        patch_choice = ttk.Radiobutton(
            merge_dialog, 
            text=f"修補版本號 (v{major}.{minor}.{int(patch)+1}) - 錯誤修復", 
            variable=version_choice, 
            value="patch"
        )
        patch_choice.pack(pady=5, padx=20, anchor=tk.W)
        
        # 確認和取消按鈕
        buttons_frame = ttk.Frame(merge_dialog)
        buttons_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        ttk.Button(
            buttons_frame, 
            text="取消", 
            command=merge_dialog.destroy
        ).pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(
            buttons_frame, 
            text="合併", 
            command=lambda: self.merge_to_main(version_choice.get(), merge_dialog)
        ).pack(side=tk.RIGHT)
    
    def get_last_version(self):
        """從version.md文件中獲取最新版本號"""
        if not os.path.exists(self.version_file_path):
            self.log("找不到version.md文件")
            return None
            
        try:
            with open(self.version_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # 使用正則表達式匹配版本行，例如 ## 1.0.6
            version_pattern = r'##\s+(\d+)\.(\d+)\.(\d+)'
            matches = re.findall(version_pattern, content)
                
            if not matches:
                self.log("未在version.md中找到版本號")
                return None
                
            # 取第一個匹配（最新版本）
            return matches[0]
                
        except Exception as e:
            self.log(f"讀取version.md時發生錯誤: {str(e)}")
            return None
    
    def merge_to_main(self, version_type, dialog):
        """合併當前分支到main分支"""
        # 關閉對話框
        dialog.destroy()
        
        # 確認進行合併
        current_branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        confirm = messagebox.askyesno(
            "確認操作", 
            f"確定要將 {current_branch} 合併到 main 分支嗎？\n\n這會先切換到main分支，再進行合併操作。"
        )
        if not confirm:
            return
            
        # 讀取上一個版本號
        last_version = self.get_last_version()
        if not last_version:
            messagebox.showerror("錯誤", "無法讀取上一個版本號")
            return
            
        major, minor, patch = [int(x) for x in last_version]
        
        # 根據選擇增加版本號
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
            
        new_version = f"v{major}.{minor}.{patch}"
        
        # 取得使用者代號
        user_code = self.user_code_var.get()
        user_code_suffix = f"[{user_code}]" if user_code else ""
        
        self.log(f"準備合併到main，新版本號: {new_version}")
        
        # 檢查當前狀態是否乾淨
        status = self.run_git_command(["git", "status", "--porcelain"], show_output=False)
        if status:
            confirm_uncommitted = messagebox.askyesno(
                "未提交變更",
                "有未提交的變更，是否先提交這些變更？"
            )
            if confirm_uncommitted:
                # 讓使用者填寫提交訊息
                commit_message = self.commit_message.get("1.0", tk.END).strip()
                if not commit_message:
                    messagebox.showwarning("警告", "提交訊息不能為空")
                    return
                
                # 添加所有文件
                self.run_git_command(["git", "add", "."])
                # 提交
                commit_result = self.run_git_command([
                    "git", "commit", "-m", f"[{new_version}] {commit_message}"
                ])
                if commit_result is None:
                    messagebox.showerror("錯誤", "提交變更失敗，無法繼續合併")
                    return
            else:
                messagebox.showinfo("取消", "請先處理未提交的變更後再合併")
                return
        
        try:
            # 1. 切換到main分支
            self.log("正在切換到main分支...")
            switch_result = self.run_git_command(["git", "checkout", "main"])
            if switch_result is None:
                raise Exception("切換到main分支失敗")
            
            # 2. 更新main分支
            self.log("正在更新main分支...")
            pull_result = self.run_git_command(["git", "pull", "origin", "main"])
            
            # 3. 合併分支
            self.log(f"正在合併 {current_branch} 到 main...")
            merge_message = f"[{new_version}]{user_code_suffix}合併版本"
            merge_result = self.run_git_command(["git", "merge", current_branch, "--no-ff", "-m", merge_message])
            if merge_result is None:
                raise Exception(f"合併 {current_branch} 到 main 失敗")
            
            # 4. 推送到遠程
            push_confirm = messagebox.askyesno("確認", "合併成功，是否推送到遠程倉庫？")
            if push_confirm:
                self.log("正在推送到遠程main分支...")
                push_result = self.run_git_command(["git", "push", "origin", "main"])
                if push_result is None:
                    messagebox.showwarning("警告", "推送到遠程倉庫失敗，但本地合併已完成")
                else:
                    messagebox.showinfo("成功", "合併完成並已推送到遠程倉庫")
            else:
                messagebox.showinfo("成功", "本地合併完成，尚未推送")
            
            # 5. 詢問是否切回原分支
            back_confirm = messagebox.askyesno("確認", f"是否切回 {current_branch} 分支？")
            if back_confirm:
                self.log(f"正在切換回 {current_branch} 分支...")
                back_result = self.run_git_command(["git", "checkout", current_branch])
                if back_result is None:
                    messagebox.showwarning("警告", f"切換回 {current_branch} 分支失敗")
            
            # 6. 刷新界面信息
            self.update_current_branch()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"合併過程中發生錯誤: {str(e)}")
            self.log(f"錯誤: {str(e)}")
            # 嘗試切回原分支
            self.run_git_command(["git", "checkout", current_branch], show_output=False)
            self.update_current_branch()
    
    def check_sync_status(self):
        """檢查本地分支與遠端分支的同步狀態"""
        # 獲取當前分支
        current_branch = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
        if not current_branch:
            messagebox.showerror("錯誤", "無法獲取當前分支")
            return
        
        # 嘗試獲取遠端分支
        self.log(f"正在檢查分支 {current_branch} 與遠端的差異...")
        
        try:
            # 確保遠端信息是最新的
            fetch_result = self.run_git_command(["git", "fetch"], show_output=False)
            
            # 檢查遠端分支是否存在
            remote_branches = self.run_git_command(["git", "branch", "-r"], show_output=False)
            remote_branch = f"origin/{current_branch}"
            
            if remote_branch not in remote_branches:
                messagebox.showinfo("訊息", f"遠端分支 {remote_branch} 不存在，請先推送此分支")
                return
            
            # 獲取本地分支與遠端分支的差異
            ahead_behind = self.run_git_command([
                "git", "rev-list", "--count", "--left-right", f"{remote_branch}...{current_branch}"
            ], show_output=False)
            
            if ahead_behind:
                # 解析結果：<behind>..<ahead>
                parts = ahead_behind.split("\t")
                if len(parts) == 2:
                    behind, ahead = int(parts[0]), int(parts[1])
                    
                    # 創建詳細信息視窗
                    self.show_sync_details(current_branch, behind, ahead)
                else:
                    messagebox.showwarning("警告", f"無法解析同步狀態: {ahead_behind}")
            else:
                messagebox.showinfo("訊息", "無法獲取分支差異信息")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"檢查同步狀態時發生錯誤: {str(e)}")
            self.log(f"錯誤: {str(e)}")
    
    def show_sync_details(self, branch, behind, ahead):
        """顯示同步詳細信息的對話框"""
        # 創建對話框
        details_dialog = tk.Toplevel(self.root)
        details_dialog.title(f"同步狀態: {branch}")
        details_dialog.geometry("600x500")
        details_dialog.transient(self.root)
        details_dialog.grab_set()
        
        # 標題
        ttk.Label(
            details_dialog, 
            text=f"分支 {branch} 與遠端的差異", 
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # 狀態顯示區
        status_frame = ttk.Frame(details_dialog)
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 落後提交數
        ttk.Label(
            status_frame,
            text=f"落後遠端: ",
            font=("Arial", 12)
        ).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        behind_lbl = ttk.Label(
            status_frame,
            text=f"{behind} 個提交",
            font=("Arial", 12, "bold"),
            foreground="red" if behind > 0 else "green"
        )
        behind_lbl.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 領先提交數
        ttk.Label(
            status_frame,
            text=f"領先遠端: ",
            font=("Arial", 12)
        ).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        ahead_lbl = ttk.Label(
            status_frame,
            text=f"{ahead} 個提交",
            font=("Arial", 12, "bold"),
            foreground="blue" if ahead > 0 else "green"
        )
        ahead_lbl.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 建議區域
        ttk.Separator(details_dialog, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        suggestion_frame = ttk.LabelFrame(details_dialog, text="建議操作")
        suggestion_frame.pack(fill=tk.X, padx=20, pady=10, expand=True)
        
        if behind > 0 and ahead > 0:
            suggestion_text = f"您需要先執行拉取(pull)操作獲取遠端更新，然後再推送(push)您的本地更改。這可能會導致合併衝突，請謹慎處理。\n\n另一選項是：先將您的變更提交到待避分支，然後更新主分支後再合併變更。"
        elif behind > 0:
            suggestion_text = f"您的本地分支落後於遠端。建議執行拉取(pull)操作以獲取最新更改。"
        elif ahead > 0:
            suggestion_text = f"您有本地更改尚未推送。建議執行推送(push)操作將您的更改同步至遠端。"
        else:
            suggestion_text = f"您的本地分支已與遠端同步，無需進一步操作。"
        
        ttk.Label(
            suggestion_frame,
            text=suggestion_text,
            wraplength=550
        ).pack(pady=10, padx=10, fill=tk.X)
        
        # 差異詳情區域
        ttk.Separator(details_dialog, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=10)
        
        diff_frame = ttk.Frame(details_dialog)
        diff_frame.pack(fill=tk.BOTH, padx=20, pady=10, expand=True)
        
        # 顯示領先的提交
        if ahead > 0:
            ttk.Label(diff_frame, text="本地領先的提交:").pack(anchor=tk.W)
            ahead_text = scrolledtext.ScrolledText(diff_frame, height=5)
            ahead_text.pack(fill=tk.BOTH, expand=True, pady=5)
            
            try:
                ahead_commits = self.run_git_command([
                    "git", "log", f"origin/{branch}..{branch}", "--pretty=format:%h %s", "--max-count=10"
                ], show_output=False)
                
                if ahead_commits:
                    ahead_text.insert(tk.END, ahead_commits)
                else:
                    ahead_text.insert(tk.END, "無法檢索提交信息")
                
                ahead_text.config(state=tk.DISABLED)
            except:
                ahead_text.insert(tk.END, "檢索提交列表時發生錯誤")
                ahead_text.config(state=tk.DISABLED)
        
        # 顯示落後的提交
        if behind > 0:
            ttk.Label(diff_frame, text="遠端領先的提交:").pack(anchor=tk.W, pady=(10, 0))
            behind_text = scrolledtext.ScrolledText(diff_frame, height=5)
            behind_text.pack(fill=tk.BOTH, expand=True, pady=5)
            
            try:
                behind_commits = self.run_git_command([
                    "git", "log", f"{branch}..origin/{branch}", "--pretty=format:%h %s", "--max-count=10"
                ], show_output=False)
                
                if behind_commits:
                    behind_text.insert(tk.END, behind_commits)
                else:
                    behind_text.insert(tk.END, "無法檢索提交信息")
                
                behind_text.config(state=tk.DISABLED)
            except:
                behind_text.insert(tk.END, "檢索提交列表時發生錯誤")
                behind_text.config(state=tk.DISABLED)
        
        # 操作按鈕
        button_frame = ttk.Frame(details_dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 當既有本地變更又落後於遠端時，提供待避分支選項
        if behind > 0 and ahead > 0:
            # 檢查是否有未提交的變更
            has_uncommitted = self.run_git_command(["git", "status", "--porcelain"], show_output=False)
            
            stash_btn = ttk.Button(
                button_frame,
                text="存入待避分支",
                command=lambda: [details_dialog.destroy(), self.create_temp_branch(branch, has_uncommitted)]
            )
            stash_btn.pack(side=tk.LEFT, padx=5)
        
        if behind > 0:
            pull_btn = ttk.Button(
                button_frame,
                text="拉取(Pull)",
                command=lambda: [details_dialog.destroy(), self.pull_changes()]
            )
            pull_btn.pack(side=tk.LEFT, padx=5)
        
        if ahead > 0:
            push_btn = ttk.Button(
                button_frame,
                text="推送(Push)",
                command=lambda: [details_dialog.destroy(), self.push_changes()]
            )
            push_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(
            button_frame,
            text="關閉",
            command=details_dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT, padx=5)

    def create_temp_branch(self, original_branch, has_uncommitted):
        """創建臨時待避分支並存儲變更"""
        # 生成臨時分支名
        timestamp = self.run_git_command(["git", "log", "-1", "--format=%at"], show_output=False)
        if not timestamp:
            timestamp = str(int(time.time()))
        
        temp_branch_name = f"stash_{original_branch}_{timestamp}"
        
        # 確認操作
        confirm = messagebox.askyesno(
            "確認",
            f"將創建待避分支 '{temp_branch_name}'\n並將您的所有變更提交到該分支。\n\n繼續操作?"
        )
        if not confirm:
            return
        
        try:
            # 創建新分支
            self.log(f"正在創建臨時待避分支: {temp_branch_name}...")
            create_result = self.run_git_command(["git", "checkout", "-b", temp_branch_name])
            
            if create_result is None:
                raise Exception("無法創建待避分支")
            
            # 如果有未提交的變更，提交它們
            if has_uncommitted:
                self.run_git_command(["git", "add", "."])
                commit_result = self.run_git_command([
                    "git", "commit", "-m", f"[STASH] 暫存變更 (從 {original_branch} 分支)"
                ])
                if commit_result is None:
                    raise Exception("無法提交變更到待避分支")
            
            # 切換回原始分支
            self.log(f"正在切換回 {original_branch} 分支...")
            checkout_result = self.run_git_command(["git", "checkout", original_branch])
            
            if checkout_result is None:
                raise Exception(f"無法切換回 {original_branch} 分支")
            
            # 更新原始分支
            self.log(f"正在更新 {original_branch} 分支...")
            pull_result = self.run_git_command(["git", "pull", "origin", original_branch])
            
            # 詢問是否要立即合併待避分支的變更
            merge_confirm = messagebox.askyesno(
                "待避完成",
                f"變更已存入待避分支 '{temp_branch_name}'，原始分支 '{original_branch}' 已更新。\n\n是否立即嘗試合併待避分支的變更?"
            )
            
            if merge_confirm:
                self.merge_stash_branch(temp_branch_name, original_branch)
            else:
                advice = (
                    f"您可以之後使用以下命令來合併待避分支的變更:\n\n"
                    f"git checkout {original_branch}\n"
                    f"git merge {temp_branch_name}\n\n"
                    f"或使用工具的「合併待避分支」功能。"
                )
                messagebox.showinfo("待避完成", advice)
            
            # 更新UI
            self.update_current_branch()
            
        except Exception as e:
            error_msg = f"待避操作失敗: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("錯誤", error_msg)
            
            # 嘗試恢復到原始分支
            self.run_git_command(["git", "checkout", original_branch], show_output=False)
            self.update_current_branch()

    def merge_stash_branch(self, stash_branch, target_branch):
        """合併待避分支到目標分支"""
        try:
            # 確保我們在目標分支上
            current = self.run_git_command(["git", "branch", "--show-current"], show_output=False)
            if current != target_branch:
                checkout_result = self.run_git_command(["git", "checkout", target_branch])
                if checkout_result is None:
                    raise Exception(f"無法切換到 {target_branch} 分支")
            
            # 執行合併
            self.log(f"正在合併 {stash_branch} 到 {target_branch}...")
            merge_result = self.run_git_command(["git", "merge", stash_branch])
            
            if merge_result is None:
                raise Exception("合併失敗，可能存在衝突")
            
            messagebox.showinfo(
                "合併成功", 
                f"待避分支 '{stash_branch}' 已成功合併到 '{target_branch}'。\n\n您可以考慮刪除待避分支。"
            )
            
            # 詢問是否刪除待避分支
            delete_confirm = messagebox.askyesno(
                "刪除待避分支",
                f"是否刪除已合併的待避分支 '{stash_branch}'?"
            )
            
            if delete_confirm:
                delete_result = self.run_git_command(["git", "branch", "-d", stash_branch])
                if delete_result is not None:
                    self.log(f"已刪除待避分支: {stash_branch}")
                else:
                    self.log(f"無法刪除分支: {stash_branch}")
            
        except Exception as e:
            error_msg = f"合併待避分支失敗: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("錯誤", error_msg)

    def read_user_codes(self):
        """從README.md讀取使用者代號"""
        try:
            if not os.path.exists(self.readme_path):
                print("找不到README.md文件")  # 使用print而非self.log以避免遞歸錯誤
                return []
                
            with open(self.readme_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 使用正則表達式匹配使用者代號行
            user_code_pattern = r'使用者代號:\s+([A-Za-z0-9])'
            matches = re.findall(user_code_pattern, content)
            
            if matches:
                try:
                    self.log(f"從README.md讀取到{len(matches)}個使用者代號")
                except:
                    print(f"從README.md讀取到{len(matches)}個使用者代號")  # 備用日誌機制
                return matches
            else:
                try:
                    self.log("未在README.md中找到使用者代號")
                except:
                    print("未在README.md中找到使用者代號")  # 備用日誌機制
                return []
                
        except Exception as e:
            error_msg = f"讀取README.md時發生錯誤: {str(e)}"
            try:
                self.log(error_msg)
            except:
                print(error_msg)  # 備用日誌機制
            return []
    
    def load_config(self):
        """讀取配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except:
                return {}
        return {}
    
    def save_config(self, config):
        """保存配置文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
        except Exception as e:
            self.log(f"保存配置文件失敗: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GitHelperApp(root)
    root.mainloop()
