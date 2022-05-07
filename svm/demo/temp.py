import copy
import math
import time

import cv2
import numpy as np


class pic:  # 表示每张指纹图
    def __init__(self, name, point, num):
        self.name = name  # 名称
        self.point = np.array(point)  # 细节点点集合
        self.num = num  # 个数
        self.p_2_d = {}  # 根据坐标转到角度
        self.createdict()

    def createdict(self):  # 创建  点坐标  2  细节点点方向   的映射
        for i in range(self.num):
            self.p_2_d[tuple(self.point[i, :2])] = self.point[i, 2]


route = ''  # 根目录
choice = ['TZ_同指.txt', 'TZ_同指200_乱序后_Data.txt', 'TZ_异指.txt']

pictures = []  # 指纹集合

'''
读取文件
'''
with open(route + choice[1], mode='r') as file:
    lines = file.readlines()
    for line in lines:
        txt = line.split(',')
        points = []
        name = txt[0]
        pictures.append(name)
        num = int(txt[1])
        txt = txt[2:]
        for i in range(int(len(txt) / 3)):
            points.append([int(txt[3 * i]), int(txt[3 * i + 1]), int(txt[3 * i + 2])])
        pictures.append(pic(name, points, num))

'''
输入：三角形的点集  type:list  size=6
输出：max(边长),周长,按照对边长度从大到小排好序的点集
'''


def get_l_per(point):
    p1 = point[:2]
    p2 = point[2:4]
    p3 = point[4:]
    d1 = math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))
    d2 = math.sqrt(pow(p1[0] - p3[0], 2) + pow(p1[1] - p3[1], 2))
    d3 = math.sqrt(pow(p3[0] - p2[0], 2) + pow(p3[1] - p2[1], 2))
    find = {d1: p3, d2: p2, d3: p1}
    d = [d1, d2, d3]
    d.sort()
    return d, d1 + d2 + d3, [find[d[0]], find[d[1]], find[d[2]]]


'''
输入:三角形的三边长
输出:γ
'''


def get_cos(d):
    a = d[0]
    b = d[1]
    c = d[2]
    cos1 = (pow(a, 2) + pow(b, 2) - pow(c, 2)) / (2 * pow(a * b, 2))
    cos2 = (pow(c, 2) + pow(b, 2) - pow(a, 2)) / (2 * pow(c * b, 2))
    cos3 = (pow(a, 2) + pow(c, 2) - pow(b, 2)) / (2 * pow(c * a, 2))
    return min(cos1, cos2, cos3)


'''
输入：PmPm+1的方向向量
输出：方向向量和细节点方向所成的夹角α
'''


def get_alpha_pp(vec):
    jiao = math.atan2(-vec[1], vec[0])  # 注意，返回的值域[-Pi,Pi]
    if jiao < 0:
        jiao += 2 * math.pi
    jiao = jiao / (math.pi) * 180
    return jiao


'''
输入：点集和指纹图对象,其中point是根据对应边排好序的点集
输出：三个α组成的列表，并将α范围控制在[0-180]
'''


def get_alpha(point, first):
    p1 = point[0]
    p2 = point[1]
    p3 = point[2]
    p_1_2 = [p2[i] - p1[i] for i in range(2)]
    p_2_3 = [p3[i] - p2[i] for i in range(2)]
    p_3_1 = [p1[i] - p3[i] for i in range(2)]

    alpha_1 = get_alpha_pp(p_1_2)
    alpha_p1 = first.p_2_d[tuple(p1)]
    alpha_1 = abs(alpha_1 - alpha_p1)
    if alpha_1 > 180:  # 在0-180之间
        alpha_1 = 360 - alpha_1

    alpha_2 = get_alpha_pp(p_2_3)
    alpha_p2 = first.p_2_d[tuple(p2)]
    alpha_2 = abs(alpha_2 - alpha_p2)
    if alpha_2 > 180:  # 在0-180之间
        alpha_2 = 360 - alpha_2

    alpha_3 = get_alpha_pp(p_3_1)
    alpha_p3 = first.p_2_d[tuple(p3)]
    alpha_3 = abs(alpha_3 - alpha_p3)
    if alpha_3 > 180:  # 在0-180之间
        alpha_3 = 360 - alpha_3

    return [alpha_1, alpha_2, alpha_3]


'''
输入：点集和指纹图对象,其中point是根据对应边排好序的点集
输出：三个β组成的列表，并将β范围控制在[0-180]
'''


def get_belta(point, first):
    p1 = point[0]
    p2 = point[1]
    p3 = point[2]

    alpha_p1 = first.p_2_d[tuple(p1)]
    alpha_p2 = first.p_2_d[tuple(p2)]
    alpha_p3 = first.p_2_d[tuple(p3)]

    alpha_1 = abs(alpha_p1 - alpha_p2)
    if alpha_1 > 180:  # 在0-180之间
        alpha_1 = 360 - alpha_1

    alpha_2 = abs(alpha_p2 - alpha_p3)
    if alpha_2 > 180:  # 在0-180之间
        alpha_2 = 360 - alpha_2

    alpha_3 = abs(alpha_p3 - alpha_p1)
    if alpha_3 > 180:  # 在0-180之间
        alpha_3 = 360 - alpha_3

    return [alpha_1, alpha_2, alpha_3]


'''
输入：三角形的三个顶点，指纹图对象
输出：三角形的特征向量
'''


def getfeature(point, first):
    l, per, sort_point = get_l_per(point)
    y = get_cos(l)
    alpha = get_alpha(sort_point, first)
    belta = get_belta(sort_point, first)
    feature = l + [y] + [per] + alpha + belta

    return feature


# 参数
class config_T:
    def __init__(self):
        # 阈值部分
        self.Thr_l = 5;
        self.Thr_y = 0.1;
        self.Thr_per = 10;
        self.Thr_alpha = 15;
        self.Thr_belta = 15;

        self.feature_path_1  # TZ_同指.txt#TZ_同指.txt转换为feature之后存储的路径
        self.feature_path_2  # TZ_同指200_乱序后_Data.txt转换为feature之后存储的路径
        self.feature_path_3  # TZ_异指.txt转换为feature之后存储的路径

        self.index_path  # 最终每一个指纹图和数据库中的指纹图的相似系数矩阵存储位置
        self.best_index_path  # 最优相似图的index的存储位子


config = config_T()

'''
输入：两个三角形的特征向量：list , list
输出：两个三角形的匹配得分
'''


def get_score(T1, T2):
    if (abs(T1[0] - T2[0]) > config.Thr_l or abs(T1[1] - T2[1]) > config.Thr_l or abs(T1[2] - T2[2]) > config.Thr_l):
        Sl = 0
        return 0
    else:
        Sl = 3 - (sum([abs(T1[i] - T2[i]) for i in range(3)]) / config.Thr_l)

    if (abs(T1[3] - T2[3]) > config.Thr_y):
        Sy = 0
        return 0
    else:
        Sy = 1 - (abs(T1[3] - T2[3]) / config.Thr_y)

    if (abs(T1[4] - T2[4]) > config.Thr_per):
        Sper = 0
        return 0
    else:
        Sper = 1 - (abs(T1[4] - T2[4]) / config.Thr_per)

    alpha_ = max([abs(T1[i] - T2[i]) for i in range(5, 8)])
    if (alpha_ > config.Thr_alpha):
        Salpha = 0
        return 0
    else:
        Salpha = 1 - alpha_ / config.Thr_alpha

    belta_ = max([abs(T1[i] - T2[i]) for i in range(8, 11)])
    if (belta_ > config.Thr_belta):
        Sbelta = 0
        return 0
    else:
        Sbelta = 1 - belta_ / config.Thr_belta

    if (Sl * Sy * Sper * Salpha * Sbelta == 0):
        return 0
    else:
        return 1 + (Sl - 1) * (Sy - 1) * (Sper - 1) * (Salpha - 1) * (Sbelta - 1)


'''
输入：两幅图的三角形网络结构，shape:n1*1 , n2*11
输出：all_triangle_1和all_triangle_2的相关系数得分
'''


def find_best(all_triangle_1, all_triangle_2):
    all_best = []
    all_score = 0
    for i in all_triangle_1:
        count = 0
        index = 0
        best_score = 0;
        best_triangle = []
        for j in all_triangle_2:
            temp = get_score(i, j)
            if (temp > best_score):
                best_triangle = j
                best_score = temp
                index = count
            count += 1
        if best_triangle != []:
            del all_triangle_2[index]
        all_best.append(best_triangle)
        all_score += best_score
    return all_best, all_score / (len(all_triangle_1) + len(all_triangle_2))


'''
输入：路径，三角形的特征矩阵
作用：将三角形的特征记录到txt避免多次运算
结构:一行表示一个图，三角形之间用空格隔开，三角形参数之间用‘，’隔开
'''


def write_2(path, feature):
    s = ''
    for k in feature:
        for j in k:
            for i in range(11):
                s += str(j[i])
                s += ','
            s += ' '
        s += '\n'
    with open(path, 'w') as f:
        f.write(s)
        f.close()


'''
输入：路径
输出：某一个文件的三角形的特征矩阵
作用：读取三角形的特征
'''


def read_2(path):
    all_feature = []
    with open(path, 'r') as file:

        for pic in file.readlines():
            triangle_txt = pic.split(' ')[:-1]
            triangle_txt = [i.split(',')[:-1] for i in triangle_txt]
            for i in range(len(triangle_txt)):
                for j in range(len(triangle_txt[i])):
                    triangle_txt[i][j] = float(triangle_txt[i][j])
            all_feature.append(triangle_txt)
    return all_feature


points_c = []
all_list = []
all_feature = []

'''
输入：三角形集合
输出：去重之后的三角形集合
'''


def only(triangleList):
    ishas = {}
    List = []
    for j in triangleList:
        tem = sorted(list(j))
        tem1 = tuple(tem)
        ishas[tem1] = ishas.get(tem1, False)
        if ishas[tem1] == False:
            List.append([int(l) for l in list(j)])
            ishas[tem1] = True
    return List


'''
输入：细节点点击
输出：增强三角剖分之后的三角集合
'''


def ET(points):
    rect = (0, 0, 1000, 1000)
    subdiv = cv2.Subdiv2D(rect)
    triangleList = []
    for i in range(len(points)):
        # EDT
        temp = points.copy()
        del temp[i]
        for p in temp:
            subdiv.insert(p)
        # 获得delaunay 三角剖分
        triangleList.extend(list(subdiv.getTriangleList()))
        # TR
        for k in range(len(temp)):
            for j in range(k + 1, len(temp)):
                triangleList.append(points[i] + temp[k] + temp[j])
    return only(triangleList)


'''
输入：int  表示对第k幅指纹图进行三角剖分
'''


def epoch(k):
    first = pictures[k]
    # 创建用于Subdiv2D 的矩形
    rect = (0, 0, 1000, 1000)
    # 创建Subdiv2D 实例
    List = []
    feature = []
    ishas = {}
    for i in range(len(first.point)):
        count = 0
        subdiv = cv2.Subdiv2D(rect);
        points = []
        for point in first.point:
            if count == i:
                count += 1
                continue
            else:
                count += 1
            points.append((int(point[0]), int(point[1])))
        # 将点插入
        for p in points:
            subdiv.insert(p)
        # 获得delaunay 三角剖分
        triangleList = list(subdiv.getTriangleList())
        for j in triangleList:
            tem = sorted(list(j))
            tem1 = tuple(tem)
            ishas[tem1] = ishas.get(tem1, False)
            if ishas[tem1] == False:
                List.append([int(l) for l in list(j)])
                feature.append(getfeature(List[-1], first))
                ishas[tem1] = True
    all_list.append(List)
    all_feature.append(feature)


'''
#两幅图的比较，寻找最优图和计算所有图的相关系数
'''


def end1(f1, f2):
    epochj = len(f2)
    epochi = len(f1)
    index = [[-1 for j in range(epochj)] for i in range(epochi)]
    best_index = [-1 for i in range(epochi)]
    for i in range(epochi):
        print(i)
        best_score = 0
        # best_triangle=[]
        for j in range(epochj):
            best_t, best_s = find_best(f1[i].copy(), f2[j].copy())  # 找到相似三角和两张图的关系系数
            index[i][j] = best_s
            if best_score < best_s:  # 如果系数比最好的还大
                best_score = best_s
                best_index[i] = j
    return best_index, index


'''
获取列表中最大的前n个数值的位置索引
'''


def getListMaxNumIndex2(num_list, topk=10):
    tmp_list = copy.deepcopy(num_list)
    tmp_list.sort()
    max_num_index = [num_list.index(one) for one in tmp_list[::-1][:topk]]
    return max_num_index


def get_relust(percentage, index):  # 获得前多少位的序号
    topk = 312
    reslut = []
    for i in range(index.shape[0]):
        reslut.append(getListMaxNumIndex2(list(index[i, :]), topk))
    return reslut


def ope(index):  # 计算比例
    count = 0
    for i in range(int(len(index) / 2)):
        count += (index[2 * i].count(10000 + 2 * i + 1)) * (index[2 * i + 1].count(10000 + 2 * i))
    print(count / len(index) * 2)


def write_pro3(index, percentage):
    index = get_relust(percentage, index)
    num = str(312)
    path = ''
    for i in range(len(index)):
        print(i)
        temp = num
        for j in range(len(index[i])):
            temp += ','
            temp += pictures[index[i][j]]
        with open(path + pictures[i + 10000] + '.txt', 'w') as f:
            f.write(temp)
            f.close()


if __name__ == '__main__':

    # 计算特征
    for i in range(len(pictures)):  # len(pictures)):
        cv = epoch(i)
    write_2(config.feature_path_2, all_feature)

    # 计算匹配

    begin = time.time()
    all_feature_3 = read_2(config.feature_path_3)
    all_feature_2 = read_2(config.feature_path_2)

    for i in all_feature_2:
        all_feature_3.append(i)
    best_index, index = end1(all_feature_2, all_feature_3)

    best_index = np.array(best_index)
    index = np.array(index)
    np.savetxt(config.best_index_path, best_index)
    np.savetxt(config.index_path_num3, index)
    end = time.time()
    print(end - begin)
