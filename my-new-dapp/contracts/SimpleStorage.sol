// SPDX-License-Identifier: MIT
pragma solidity ^0.5.16;
contract SimpleStorage {
    // 建議改用 uint256 以語義更明確
    uint256 private storedData;

    // 設置新的值
    function set(uint256 x) public {
        storedData = x;
    }

    // 取得目前的值
    function get() public view returns (uint256) {
        return storedData;
    }
}
