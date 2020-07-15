import numpy as np
import csv
import matplotlib.pyplot as plt

""""对男女声音进行辨别"""


def load_data_set(file_name,n):
    """
    :param file_name: 文件名字
    :return

    train_mat：离散化的训练数据集
    train_classes： 训练数据集所属的分类
    test_mat：离散化的测试数据集
    test_classes：测试数据集所述的分类
    label_name：特征的名称
    """
    data_mat = []
    with open(file_name) as file_obj:
        voice_reader = csv.DictReader(file_obj)     #voice_reader是一个dict数组，每一个元素都是一个dict,一个dict包含一组数据
        list_class = []
        # 文件头
        label_name = list(voice_reader.fieldnames)  #顾名思义，label_name是一个特征名字数组
        num = len(label_name) - 1

        for line in voice_reader.reader:
            data_mat.append(line[:num])              #data_mat是一个列表，每一个元素也是一个列表，元素列表中是一组数据的字符型时
            gender = 1 if line[-1] == 'male' else 0  #male表示为1，female表示为0
            list_class.append(gender)                #list_class是标签列表

        # 求每一个特征的平均值 
        data_mat = np.array(data_mat).astype(float)        #将数字字符串转化为浮点数
        count_vector = np.count_nonzero(data_mat, axis=0)  #统计每一列‘非零元素’个数
        sum_vector = np.sum(data_mat, axis=0)              #每列求和
        mean_vector = sum_vector / count_vector            #每列，即每一个特征值的平均数（不含0元素）

        # 数据缺失的地方 用 平均值填充
        for row in range(len(data_mat)):
            for col in range(num):
                if data_mat[row][col] == 0.0:
                    data_mat[row][col] = mean_vector[col]  #0元素位置用对应平均值填充

        # 将数据连续型的特征值离散化处理
        min_vector = data_mat.min(axis=0)
        max_vector = data_mat.max(axis=0)
        diff_vector = max_vector - min_vector
        diff_vector /= n

        new_data_set = []                          #用于存放离散化后的数据
        for i in range(len(data_mat)):
            line = np.array((data_mat[i] - min_vector) / diff_vector).astype(int)
            new_data_set.append(line)

        # 随机划分数据集为训练集 和 测试集
        test_set = list(range(len(new_data_set)))
        train_set = []
        for i in range(2200):
            random_index = int(np.random.uniform(0, len(test_set)))
            train_set.append(test_set[random_index])     #保存训练数据集编号
            del test_set[random_index]                   #保存测试数据集的编号

        # 训练数据集
        train_mat = []
        train_classes = []
        for index in train_set:
            train_mat.append(new_data_set[index])
            train_classes.append(list_class[index])

        # 测试数据集
        test_mat = []
        test_classes = []
        for index in test_set:
            test_mat.append(new_data_set[index])
            test_classes.append(list_class[index])

    return train_mat, train_classes, test_mat, test_classes, label_name    #数据集是列表类型，标签集是列表类型,label_name是各个特征名字列表


def native_bayes(train_matrix, list_classes,n):
    """
    :param train_matrix: 训练样本矩阵
    :param list_classes: 训练样本分类向量
    :n: 数据的离散度
    :return:p_1_class 任一样本分类为1的概率  p_feature,p_1_feature 分别为给定类别的情况下所以特征所有取值的概率
    """
    # 训练样本个数
    num_train_data = len(train_matrix)  
    # 每个样本特征数
    num_feature = len(train_matrix[0])
    # 分类为1的样本占比
    p_1_class = sum(list_classes) / float(num_train_data)

    list_classes_1 = []
    train_data_1 = []

    for i in list(range(num_train_data)):
        if list_classes[i] == 1:
            #获得标签为1的数据位置
            list_classes_1.append(i)                     
            #获得标签为1的数据集
            train_data_1.append(train_matrix[i])         

    # 分类为1 情况下的各特征的概率
    train_data_1 = np.matrix(train_data_1)
    p_1_feature = {}
    for i in list(range(num_feature)):
        #将每一列的特征值提取出来单独作为一个数组(矩阵的切片，压缩)
        feature_values = np.array(train_data_1[:, i]).flatten()         
        # 拉普拉斯平滑，避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))   
        p_1_feature[i] = p
     

    # 所有分类下的各特征的概率
    p_feature = {}
    train_matrix = np.matrix(train_matrix)
    for i in list(range(num_feature)):
        feature_values = np.array(train_matrix[:, i]).flatten()
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))
        #最后得到的结果与上述相似，知识概率为所有分类中的概率
        p_feature[i] = p

    return p_feature, p_1_feature, p_1_class


def classify_bayes(test_vector, p_feature, p_1_feature, p_1_class,weight):
    """
    :param test_vector: 要分类的测试向量
    :param p_feature: 所有分类的情况下特征所有取值的概率
    :param p_1_feature: 类别为1的情况下所有特征所有取值的概率
    :param p_1_class: 任一样本分类为1的概率
    :return: 1 表示男性 0 表示女性
    """
    # 计算每个分类的概率(概率相乘取对数 = 概率各自对数相加)
    sum = 0.0
    for i in list(range(len(test_vector))):
        #先验条件概率相乘，因为取了log所以直接相加
        sum += p_1_feature[i][test_vector[i]]
        #除以各个离散值的出现概率，也就是减去log值
        sum -= p_feature[i][test_vector[i]]
        # 只保留影响力大的
        if i not in weight: 
            sum -= p_1_feature[i][test_vector[i]]
            sum += p_feature[i][test_vector[i]]
    #最后乘以标签为1的概率获得后验概率，也就是加log值
    p1 = sum + np.log(p_1_class)

    if p1 > np.log(0.5):
        return 1
    else:
        return 0


def test_bayes(n,weight):
    file_name = 'voice.csv'
    train_mat, train_classes, test_mat, test_classes, label_name = load_data_set(file_name,n-1)

    p_feature, p_1_feature, p_1_class = native_bayes(train_mat, train_classes,n)
    #总正确率
    count = 0.0
    correct_count = 0.0
    #男性正确率
    male_count = 0.0
    male_correct_count = 0.0
    #女性正确率
    female_count = 0.0
    female_correct_count = 0.0

    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        result = classify_bayes(test_vector, p_feature, p_1_feature, p_1_class,weight)
        if result == test_classes[i]:
            #统计总正确率
            correct_count += 1
            #统计男性正确率
            if test_classes[i]==1:
                male_count += 1
                male_correct_count += 1
            #统计女性正确率
            else:
                female_count += 1
                female_correct_count += 1
        #统计错误率
        else:
            if test_classes[i]==1:
                male_count += 1
            else:
                female_count += 1
        count += 1

    return (correct_count / count),(male_correct_count / male_count),(female_correct_count / female_count)

if __name__ == '__main__':
#计算100次取平均值
    sum1=0.0
    male_sum1=0.0
    female_sum1=0.0
    sum2=0.0
    male_sum2=0.0
    female_sum2=0.0
    for i in range(100):
        sum1,male_sum1,female_sum1=test_bayes(22,[1,3,5,12])
        sum2+=sum1
        male_sum2+=male_sum1
        female_sum2+=female_sum1

    print('总正确率:%.3f%%'%(round((sum2/100)*100.0,3)))
    print('男性正确率:%.3f%%'%(round((male_sum2/100)*100.0,3)))
    print('女性正确率:%.3f%%'%(round((female_sum2/100)*100.0,3)))
    print('男性错误率:%.3f%%'%(round((1-male_sum2/100)*100.0,3)))
    print('女性错误率:%.3f%%'%(round((1-female_sum2/100)*100.0,3)))