const SimpleStorage = artifacts.require('./SimpleStorage.sol');
const Accounts = artifacts.require("Accounts");
const CarbonCreditToken = artifacts.require("CarbonCreditToken");

module.exports = function (deployer) {
  deployer.deploy(SimpleStorage);
  deployer.deploy(Accounts);
  deployer.deploy(CarbonCreditToken);
}
