// SPDX-License-Identifier: MIT

pragma solidity ^0.5.16;
contract Accounts {
    struct Account {
        string name;
        uint256 balance;
    }

    mapping(address => Account) private accounts;

    function createAccount(string memory _name, uint256 _balance) public {
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(accounts[msg.sender].balance == 0, "Account already exists");
        accounts[msg.sender] = Account(_name, _balance);
    }

    function getAccount(address _address) public view returns (string memory, uint256) {
        require(bytes(accounts[_address].name).length > 0, "Account does not exist");
        return (accounts[_address].name, accounts[_address].balance);
    }

    function getMyAccount() public view returns (string memory, uint256) {
        require(bytes(accounts[msg.sender].name).length > 0, "Account does not exist");
        return (accounts[msg.sender].name, accounts[msg.sender].balance);
    }
}