''' classification and regression tree (CART)
    知识点: 递归, 基尼系数
'''
def choose_best_feature(dataset):
    numFeatures = len(dataset[0]) - 1
    if numFeatures == 1:
        return 0
    bestGini = 1
    index_of_best_feature = -1
    for i in range(numFeatures):
        uniqueVals = set(example[i] for example in dataset)
        Gini = {}
        for value in uniqueVals:
            sub_dataset1, sub_dataset2 = split_dataset(dataset,i,value)
            prob1 = len(sub_dataset1) / float(len(dataset))
            prob2 = len(sub_dataset2) / float(len(dataset))
            Gini_of_sub_dataset1 = calcGini(sub_dataset1)
            Gini_of_sub_dataset2 = calcGini(sub_dataset2)
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                index_of_best_feature = i
                best_split_point = value
    return index_of_best_feature, best_split_point


def create_decision_tree(dataset, features):
    label_list = [example[-1] for example in dataset]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    if len(dataset[0]) == 1:
        return find_label(label_list)
    index_of_best_feature, best_split_point = choose_best_feature(dataset)
    # 得到最佳特征
    best_feature = features[index_of_best_feature]
    # 初始化决策树
    decision_tree = {best_feature: {}}
    # 使用过当前最佳特征后将其删去
    del(features[index_of_best_feature])
    # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    sub_labels = features[:]
    # 递归调用create_decision_tree去生成新节点
    # 生成由最优切分点划分出来的二分子集
    sub_dataset1, sub_dataset2 = split_dataset(dataset,index_of_best_feature,best_split_point)
    # 构造左子树
    decision_tree[best_feature][best_split_point] = create_decision_tree(sub_dataset1, sub_labels)
    # 构造右子树
    decision_tree[best_feature]['others'] = create_decision_tree(sub_dataset2, sub_labels)
    return decision_tree
