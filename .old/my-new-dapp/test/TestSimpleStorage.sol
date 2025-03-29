// SPDX-License-Identifier: MIT
pragma solidity ^0.5.16;

import "truffle/Assert.sol";
import "truffle/DeployedAddresses.sol";
import "../contracts/SimpleStorage.sol";

contract TestSimpleStorage {
    function testItStoresAValue() public {
        // 取得已部署的 SimpleStorage 合約地址
        SimpleStorage simpleStorage = SimpleStorage(DeployedAddresses.SimpleStorage());

        // 呼叫 set(89)
        simpleStorage.set(89);

        // 預期測試值
        uint256 expected = 89;

        // 透過 Truffle 的 Assert 函式來檢查合約回傳值是否與預期相符
        Assert.equal(simpleStorage.get(), expected, "It should store the value 89.");
    }
}
