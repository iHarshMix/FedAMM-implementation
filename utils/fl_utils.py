
def avg_local_weights(w1,w2,w3,w4,m):
    # 初始化一个空的字典来存储聚合后的权重
    aggregated_weights = {}
    # 获取所有客户端模型的权重
    for key in w1.keys():
        # 初始化一个变量来存储当前键对应的权重之和
        sum_weight = 0
        # 遍历所有客户端模型
        for i, client_weight in enumerate([w1, w2, w3, w4]):
            # 将当前键对应的权重累加到sum_weight中
            #sum_weight += client_weight[key].data.cpu()*m[i]
            sum_weight += client_weight[key].data.cpu()
        # 计算平均权重
        avg_weight = sum_weight / 4
        #avg_weight = sum_weight
        # 将平均权重存储到聚合后的权重字典中
        aggregated_weights[key] = avg_weight
    # 返回聚合后的权重
    return aggregated_weights

def avg_encoder_weights(w1,w2,w3,w4,client_modal_weight):
    # 初始化一个空的字典来存储聚合后的权重
    aggregated_weights = {}
    # 获取所有客户端模型的权重
    for key in w1.keys():
        # 初始化一个变量来存储当前键对应的权重之和
        sum_weight = 0
        # 遍历所有客户端模型
        for i, client_weight in enumerate([w1, w2, w3, w4]):
            # 将当前键对应的权重累加到sum_weight中
            sum_weight += client_weight[key].data.cpu()*client_modal_weight[i]
        # 计算平均权重
        avg_weight = sum_weight
        # 将平均权重存储到聚合后的权重字典中
        aggregated_weights[key] = avg_weight
    # 返回聚合后的权重
    return aggregated_weights

    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample
    
def avg_imputer_weights(w1, w2, w3, w4, imputer_weights):
    """
    Aggregate imputer parameters weighted by (available x missing) pair counts.
    imputer_weights: list of 4 floats — one per client, summing to 1.
    """
    aggregated = {}
    for key in w1.keys():
        sum_weight = 0
        for i, client_w in enumerate([w1, w2, w3, w4]):
            sum_weight += client_w[key].data.cpu() * imputer_weights[i]
        aggregated[key] = sum_weight
    return aggregated