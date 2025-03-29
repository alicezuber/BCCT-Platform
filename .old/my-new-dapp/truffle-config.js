module.exports = {
  networks: {
    development: {
      host: "25.3.19.183", // 本地主機
      port: 7545,        // Ganache 默認端口
      network_id: "5777",   // 匹配任何 network id
    }
  },
  solc: {
    version: "0.5.16",
    // Turns on the Solidity optimizer. For development the optimizer's
    // quite helpful, just remember to be careful, and potentially turn it
    // off, for live deployment and/or audit time. For more information,
    // see the Truffle 4.0.0 release notes.
    //
    // https://github.com/trufflesuite/truffle/releases/tag/v4.0.0
    optimizer: {
      enabled: true,
      runs: 200
    }
  }
}
