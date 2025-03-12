const getContractInstance = async (web3, contractDefinition) => {
  // 1. 取得網路 ID
  const networkId = await web3.eth.net.getId()

  // 2. 從 contractDefinition 裡讀取對應 networkId 的部署位址
  const deployedAddress = contractDefinition.networks[networkId].address

  // 3. 用該位址和 ABI，實例化出一個 web3.eth.Contract
  const instance = new web3.eth.Contract(
    contractDefinition.abi,
    deployedAddress
  )
  return instance
}

export default getContractInstance