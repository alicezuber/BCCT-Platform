pragma solidity 0.5.16;

// 安全數學運算庫，防止數值溢出
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        return a - b;
    }
}

// 碳權代幣合約，實作 ERC20 部分標準功能
contract CarbonCreditToken{
    using SafeMath for uint256;
    
    // 代幣資訊
    string public name = "Carbon Credit Token";
    string public symbol = "CCT";
    uint8 public decimals = 18;
    
    // 代幣總供應量
    uint256 public totalSupply;
    
    // 每個地址的餘額
    mapping (address => uint256) public balanceOf;
    // 每個地址的授權額度 (owner => (spender => amount))
    mapping (address => mapping (address => uint256)) public allowance;
    
    // 事件：轉帳
    event Transfer(address indexed from, address indexed to, uint256 value);
    // 事件：授權
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    // 建構子（初始供應量預設為 0）
    constructor() public {
        totalSupply = 0;
    }
    
    // 轉帳函數：從 msg.sender 將代幣轉至指定地址
    function transfer(address to, uint256 value) public returns (bool success) {
        require(to != address(0), "無效地址");
        require(balanceOf[msg.sender] >= value, "餘額不足");
        
        balanceOf[msg.sender] = balanceOf[msg.sender].sub(value);
        balanceOf[to] = balanceOf[to].add(value);
        
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    // 授權函數：允許 spender 從 msg.sender 帳戶轉出指定數量的代幣
    function approve(address spender, uint256 value) public returns (bool success) {
        require(spender != address(0), "無效地址");
        
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    
    // 從授權帳戶轉帳：由授權人 (msg.sender) 從 sender 帳戶將代幣轉出給 recipient
    function transferFrom(address sender, address recipient, uint256 value) public returns (bool success) {
        require(recipient != address(0), "無效地址");
        require(balanceOf[sender] >= value, "餘額不足");
        require(allowance[sender][msg.sender] >= value, "授權額度不足");
        
        balanceOf[sender] = balanceOf[sender].sub(value);
        balanceOf[recipient] = balanceOf[recipient].add(value);
        allowance[sender][msg.sender] = allowance[sender][msg.sender].sub(value);
        
        emit Transfer(sender, recipient, value);
        return true;
    }
    
    // Mint 函數：使用者可以輸入欲增加的代幣數量，將代幣發行給自己
    // （注意：這樣設計意味著任何人皆可無限制 mint，僅適用於示範用途）
    function mint(uint256 amount) public {
        balanceOf[msg.sender] = balanceOf[msg.sender].add(amount);
        totalSupply = totalSupply.add(amount);
        
        // 依 ERC20 標準，從 0 地址轉出代表鑄幣動作
        emit Transfer(address(0), msg.sender, amount);
    }

}
