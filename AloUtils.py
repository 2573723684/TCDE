import random
def savePopulation_Scores(all_population,current_population,all_scores,current_scores):
    x = []
    num = len(current_scores)
    for i in range(len(current_scores)):
        for j in range(len(all_scores)):
            if all_population[j][0] == current_population[i][0] and all_population[j][1] == current_population[i][1] and all_population[j][2] == current_population[i][2]:
                num = num - 1
                x.append(i)
                break
    tempPopulatiaon = np.empty((num + len(all_scores), len(current_population[0])), dtype=float)
    tempScores = np.empty((num + len(all_scores), 1), dtype=float)

    for i in range(len(all_scores)):
        tempPopulatiaon[i][0] = all_population[i][0]
        tempPopulatiaon[i][1] = all_population[i][1]
        tempPopulatiaon[i][2] = all_population[i][2]
        tempScores[i] = all_scores[i]
    count = len(all_scores)
    for i in range(len(current_scores)):
        if i not in x:
            tempPopulatiaon[count][0] = current_population[i][0]
            tempPopulatiaon[count][1] = current_population[i][1]
            tempPopulatiaon[count][2] = current_population[i][2]
            tempScores[count] = current_scores[i]
            count = count + 1
    return tempPopulatiaon, tempScores

from cec2013.cec2013 import *
def cec2013main(poplation,func_num):
    scores = np.empty((len(poplation),1))
    func = func_num
    f = CEC2013(func)
    for i in range(len(poplation)):
        scores[i] = f.evaluate(poplation[i])
    return scores

def desort(arr):
    indexes = np.empty((len(arr), 1), dtype=int)
    indexs = np.argsort(arr,axis=0)
    count = len(arr) -1
    for i in range(len(indexs)):
        indexes[count] = indexs[i]
        count = count - 1
    return indexes

from pykrige.ok import OrdinaryKriging
from pykrige.uk3d import UniversalKriging3D
from pykrige.ok3d import OrdinaryKriging3D
#3维克里金模型
def mulKringingModel(population,scores, params):
    data1 = np.empty((len(population),1))
    data2 = np.empty((len(population),1))
    data3 = np.empty((len(population),1))
    for i in range(len(population)):
        data1[i] = population[i][0]
        data2[i] = population[i][1]
        data3[i] = population[i][2]

    ok3d = UniversalKriging3D(data1, data2, data3, scores[:,0], variogram_model="linear")

    range_step = abs(params[0]-params[1]) / 50

    gridx = np.arange(params[0], params[1] + range_step, range_step)
    gridy = np.arange(params[2], params[3] + range_step, range_step)
    gridz = np.arange(params[4], params[5] + range_step, range_step)
    # 3维克里金模型生成栅格数据
    k3d1, ss3d = ok3d.execute("grid", gridx, gridy, gridz)

    m1,m2 = select3DMultimodal2(population, scores, params[0], params[1], params[2], params[3], params[4], params[5], range_step,
                       k3d1)
    return m1, m2


def kringingModel(population,scores,x_lower,x_upper,y_lower,y_upper,range_step):

    data = np.empty((len(population),len(population[0])+1),dtype=float)
    for i in range(len(population)):
        data[i][0] = population[i][0]
        data[i][1] = population[i][1]
        data[i][2] = scores[i]
    gridx = np.arange(x_lower, x_upper, range_step)  # 三个参数的意思：范围0.0 - 0.6 ，每隔0.1划分一个网格
    gridy = np.arange(y_lower, y_upper, range_step)
    ok3d = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="linear")

    k3d1, ss3d = ok3d.execute("grid", gridx, gridy)

    MulimodalPop,MulimodalS = selectMultimodal(population,scores,x_lower,x_upper,y_lower,y_upper,range_step,k3d1)
    return MulimodalPop,MulimodalS,k3d1


def selectMultimodal(population,scores,x_lower,x_upper,y_lower,y_upper,range_step,k3d1):

    temp = []
    for i in range(len(k3d1) - 2):
        for j in range(len(k3d1) - 2):
            locBest = [1000, 1000, -100000]
            for p in range(len(population)):
                if population[p][0] >= x_lower + i * range_step and population[p][0] < x_lower + (i + 1) * range_step and population[p][0] < x_upper and population[p][1] < y_upper and population[p][1] >= y_lower + j * range_step and population[p][1] < y_lower + (j + 1) * range_step:
                    if scores[p] >= locBest[2]:
                        locBest[0] = population[p][0]
                        locBest[1] = population[p][1]
                        locBest[2] = scores[p]
            if k3d1[i][j] < locBest[2] and k3d1[i + 1][j] < locBest[2] and k3d1[i][j + 1] < locBest[2] and k3d1[i + 1][j + 1] < locBest[2] and locBest[2] != -100000:
                temp.append(locBest[0])
                temp.append(locBest[1])
                temp.append(locBest[2])

    MulimodalPop = np.empty((int(len(temp)/3),len(population[0])))
    MulimodalS = np.empty((int(len(temp)/3),1))
    count = 0
    for i in range(len(temp)):
        if (i % 3 == 0) and i > 0:
            count = count + 1
        if(i % 3 == 0):
            MulimodalPop[count][0] = temp[i]
        if(i % 3 == 1):
            MulimodalPop[count][1] = temp[i]
        if(i % 3 == 2):
            MulimodalS[count] = temp[i]

    return MulimodalPop,MulimodalS

def select3DMultimodal(population, scores, x_lower, x_upper, y_lower, y_upper,z_lower, z_upper,range_step, k3d1):
    temp = []
    for i in range(len(k3d1) - 2):
        for j in range(len(k3d1) - 2):
            for x in range(len(k3d1) - 2):
                locBest = [1000, 1000, 1000,-100000]
                for p in range(len(population)):
                    if population[p][0] >= x_lower + i * range_step and population[p][0] < x_lower + (i + 1) * range_step and \
                            population[p][0] < x_upper and population[p][1] < y_upper and population[p][
                        1] >= y_lower + j * range_step and population[p][1] < y_lower + (j + 1) * range_step and population[p][2] < z_upper and population[p][2] >= z_lower + x * range_step and population[p][2] < z_lower + (x+1) * range_step:
                        if scores[p] >= locBest[3]:
                            locBest[0] = population[p][0]
                            locBest[1] = population[p][1]
                            locBest[2] = population[p][2]
                            locBest[3] = scores[p]
                if k3d1[i][j][x] < locBest[3] and k3d1[i + 1][j][x] < locBest[3] and k3d1[i][j + 1][x] < locBest[3] and \
                        k3d1[i + 1][j + 1][x] < locBest[3] and locBest[3] != -100000 and k3d1[i][j][x + 1] < locBest[3] and k3d1[i+1][j+1][x+1] < locBest[3] and k3d1[i][j+1][x + 1] < locBest[3] and k3d1[i + 1][j][x + 1] < locBest[3]:
                    temp.append(locBest[0])
                    temp.append(locBest[1])
                    temp.append(locBest[2])
                    temp.append(locBest[3])
    MulimodalPop = np.empty((int(len(temp) / 4), len(population[0])))
    MulimodalS = np.empty((int(len(temp) / 4), 1))
    count = 0
    for i in range(len(temp)):
        if (i % 4 == 0) and i > 0:
            count = count + 1
        if (i % 4 == 0):
            MulimodalPop[count][0] = temp[i]
        if (i % 4 == 1):
            MulimodalPop[count][1] = temp[i]
        if (i % 4 == 2):
            MulimodalPop[count][2] = temp[i]
        if (i % 4 == 3):
            MulimodalS[count] = temp[i]
    return MulimodalPop, MulimodalS
# 2023 8 9
def select3DMultimodal1(population, scores, x_lower, x_upper, y_lower, y_upper,z_lower, z_upper,range_step, k3d1):
    MulimodalPop = np.empty((0,len(population[0])))
    MulimodalS = np.empty((0,1))
    k_length = len(k3d1) - 2
    for p in range(len(population)):
        founded = 0
        for i in range(k_length):
            if founded == 1:
                break
            if x_lower + i * range_step <= population[p][0] < x_lower + (i + 1) * range_step and population[p][0] < x_upper:
                for j in range(k_length):
                    if founded == 1:
                        break
                    if population[p][1] < y_upper and y_lower + j * range_step <= population[p][1]  < y_lower + (j + 1) * range_step:
                        for x in range(k_length):
                            if founded == 1:
                                break
                            if x_lower + i * range_step <= population[p][0] < x_lower + (i + 1) * range_step and population[p][0] < x_upper and population[p][1] < y_upper and y_lower + j * range_step <= population[p][1] < y_lower + (j + 1) * range_step and population[p][2] < z_upper and  z_lower + x * range_step <= population[p][2] < z_lower + (x + 1) * range_step:
                                founded = 1
                                if k3d1[i][j][x] < scores[p] and k3d1[i + 1][j][x] < scores[p] and k3d1[i][j + 1][x] < scores[p]  and k3d1[i + 1][j + 1][x] < scores[p]  and k3d1[i][j][x + 1] < scores[p]  and k3d1[i+1][j+1][x+1] < scores[p]  and k3d1[i][j+1][x + 1] < scores[p]  and k3d1[i + 1][j][x + 1] < scores[p] :
                                    MulimodalPop = np.vstack((MulimodalPop,np.array([[population[p][0],population[p][1],population[p][2]]])))

                                    MulimodalS = np.vstack((MulimodalS,np.array([scores[p]])))
    return MulimodalPop, MulimodalS

        #2023 8 15
def select3DMultimodal2(population, scores, x_lower, x_upper, y_lower, y_upper,z_lower, z_upper,range_step, k3d1):
    num_individuals, num_dimensions = population.shape
    num_steps = int((x_upper - x_lower) / range_step) + 1
    MulimodalPop = np.empty((0, len(population[0])))
    MulimodalS = np.empty((0, 1))
    for ind in range(num_individuals):
        index = []
        for dim in range(num_dimensions):
            dim_values = population[ind, dim]
            lower_idx = int((dim_values - x_lower) // range_step)
            upper_idx = lower_idx + 1
            lower_idx = max(0, min(lower_idx, num_steps - 1))
            upper_idx = max(0, min(upper_idx, num_steps - 1))
            index.append(upper_idx)
            index.append(lower_idx)
        if k3d1[index[0]] [index[2]] [index[4]] < scores[ind] and k3d1[index[1]] [index[2]] [index[4]] < scores[ind] and k3d1[index[0]] [index[3]] [index[4]] < scores[ind] and k3d1[index[1]] [index[3]] [index[4]] < scores[ind] and k3d1[index[0]] [index[2]] [index[5]] < scores[ind] and k3d1[index[1]] [index[3]] [index[5]] < scores[ind] and k3d1[index[0]] [index[3]] [index[5]] < scores[ind] and k3d1[index[1]] [index[2]] [index[5]] < scores[ind]:
            MulimodalPop = np.vstack((MulimodalPop,np.array([[population[ind][0],population[ind][1],population[ind][2]]])))
            MulimodalS = np.vstack((MulimodalS, np.array([scores[ind]])))
    return MulimodalPop,MulimodalS

import kriging
def Min(all_population, func_num, all_scores ,param_bound):

    allMulimodalPop = np.empty((0,len(all_population[0])))

    allMulimodalS = np.empty((0,1))

    length = (param_bound[1] - param_bound[0]) / 2

    for l in range(2):
        x = []
        xs = []
        x_lower = param_bound[0] + l * length
        x_upper = param_bound[0] + (l + 1) * length
        for pop in range(len(all_population)):
            if x_lower <= all_population[pop][0] < x_upper:
                x.append(all_population[pop])
                xs.append(all_scores[pop])
        for w in range(2):
            xy = []
            xys = []
            y_lower = param_bound[0] + w * length
            y_upper = param_bound[0] + (w + 1) * length
            for pop in range(len(x)):
                if y_lower <= x[pop][1] < y_upper:
                    xy.append(x[pop])
                    xys.append(xs[pop])
            for h in range(2):
                xyz = []
                xyzs = []
                z_lower = param_bound[0] + h * length
                z_upper = param_bound[0] + (h + 1) * length
                for pop in range(len(xy)):
                    if z_lower <= xy[pop][2] < z_upper:
                        xyz.append(xy[pop])
                        xyzs.append(xys[pop])
                params = [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]

                # 克里金输入不能为空
                if len(np.array(xyz)) > 4:

                    MulimodalPop, MulimodalS = mulKringingModel(np.array(xyz), np.array(xyzs), params)
                    allMulimodalPop = np.vstack((allMulimodalPop,MulimodalPop))
                    allMulimodalS = np.vstack((allMulimodalS,MulimodalS))
    return allMulimodalPop,allMulimodalS


def dataSelect(population,scores,param_bound,func_num,selectParam,selectParamStart,NR):
    sorted_index = desort(scores)
    sorted_pop = np.empty((len(population),len(population[0])))
    sorted_s = np.empty((len(population),1))
    for i in range(len(population)):
        for j in range(len(population[0])):
            sorted_pop[i][j] = population[int(sorted_index[i])][j]
        sorted_s[i] = scores[int(sorted_index[i])]

    selectMountCount = int(len(scores) * (selectParam - selectParamStart))

    selectMountPop = np.empty((selectMountCount,len(population[0])))
    selectMountS = np.empty((selectMountCount, 1))

    for i in range(len(selectMountPop)):
        for j in range(len(selectMountPop[0])):
            selectMountPop[i][j] = sorted_pop[int(i + len(population) * selectParamStart)][j]
        selectMountS[i] = sorted_s[int(i + len(population) * selectParamStart)]
    return selectMountPop,selectMountS


from sklearn.cluster import DBSCAN

def dbscn(population,radious,min_points,scores):
    X_blobs = population
    dbscan = DBSCAN(eps=radious, min_samples=min_points)
    labels = dbscan.fit_predict(X_blobs)
    clusterPop = []
    clusterS = []
    for i in range(len(population)):
        res = []
        res1 = []
        if labels[i] != -1:
            for j in range(len(population[0])):
                res.append(population[i][j])
            res1.append(scores[i])
            clusterPop.append(res)
            clusterS.append(res1)
    return labels,np.array(clusterPop),np.array(clusterS)

def howManyGlobal(accuracy,func_num,Pop):
    f = CEC2013(func_num)
    count, seeds = how_many_goptima(Pop, f, accuracy)
    print(f"In the current population there exist {count} global optimizers.")
    # print(f"Global optimizers: {seeds}")
    return count


def Odistance(Ts,pop,index):
    if len(Ts) == 0:
        return 1
    count = 0
    for i in range(len(Ts)):
        sum = 0
        for j in range(len(Ts[0]) - 1):
            sum += (Ts[i][j] - pop[index][j]) ** 2
        distance = sum ** 0.5
        if distance > Ts[i][-1]:
            count += 1
    if count == len(Ts):
        return 1
    else:
        return 0

def Odistance1(Ts,pop,index):
    for i in range(len(Ts)):
        sum = 0
        for j in range(len(Ts[0]) - 1):
            sum += (Ts[i][j] - pop[index][j]) ** 2
        distance = sum ** 0.5
        if distance < Ts[i][-1]:
            return 1
    return 0


def newbaseClusterSearch(clusters,dbscn_population,dbscn_scores):
    pass

def baseClusterSearch(clusters,dbscn_population,dbscn_scores,bestpop):

    temp = np.empty((len(dbscn_population), len(dbscn_population[0]) + 2))
    for i in range(len(dbscn_population)):
        for j in range(len(dbscn_population[0])):
            temp[i][j] = dbscn_population[i][j]
        temp[i][-2] = dbscn_scores[i]
        temp[i][-1] = clusters[i]

    parameter = np.empty((max(clusters) + 1,len(dbscn_population[0]) + 1))

    parameterlast = np.empty((max(clusters) + 1, len(dbscn_population[0]) + 1))

    for i in range(len(parameterlast)):
        parameterlast[i][-1] = 999999999
        parameter[i][-1] = -999999999

    Multiple_nodes_num = max(clusters) + 1

    for Multiple_node in range(Multiple_nodes_num):

        for i in range(len(temp)):
            if Multiple_node == temp[i][-1]:
                if temp[i][-2] >= parameter[Multiple_node][-1]:
                    for j in range(len(parameter[0])):
                        parameter[Multiple_node][j] = temp[i][j]
                if temp[i][-2] <= parameterlast[Multiple_node][-1]:
                    for j in range(len(parameterlast[0])):
                        parameterlast[Multiple_node][j] = temp[i][j]

    selectround = np.empty((len(parameter),1))

    point1 = parameter[:, :3]
    point2= parameterlast[:, :3]
    for i in range(len(selectround)):
        selectround[i] = np.linalg.norm(point1[i] - point2[i])
        # if 3 < parameter[i][0]< 5 or 3 < parameter[i][1] < 5 or 3 < parameter[i][2] < 5:
        #     selectround[i] = np.linalg.norm(point1[i] - point2[i]) // 2
        # elif parameter[i][0]< 3 or parameter[i][1] < 3 or parameter[i][2] < 3:
        #     selectround[i] = np.linalg.norm(point1[i] - point2[i]) // 4
        # else:
        #     selectround[i] = np.linalg.norm(point1[i] - point2[i])
        #2023 8 19 /2
    selectBound = np.hstack((np.hstack((parameter, selectround)),parameterlast))

    return selectBound

def selectBoundInit(x,y,round,population_size,func_num):
    pop = np.empty((population_size,2))
    for i in range(len(pop)):
        pop[i][0] = np.random.uniform(x - round,x + round,1)
        pop[i][1] = np.random.uniform(y - round,y + round,1)
        while (((pop[i][0] - x) ** 2 + (pop[i][1] - y) ** 2) ** 0.5) > round :
            pop[i][0] = np.random.uniform(x - round, x + round, 1)
            pop[i][1] = np.random.uniform(y - round, y + round, 1)
    s = cec2013main(pop,func_num)
    return pop,s

def generate_points_in_sphere(center, radius, num_points,params):
    lower = params[0]
    upper = params[1]

    radii = np.power(np.random.uniform(0, 1, num_points), 1 / 3) * radius
    theta = np.random.uniform(0, np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    x = center[0] + radii * np.sin(theta) * np.cos(phi)
    y = center[1] + radii * np.sin(theta) * np.sin(phi)
    z = center[2] + radii * np.cos(theta)
    while x < lower or x > upper or y < lower or y > upper or z < lower or z > upper:
        r = np.power(np.random.uniform(0, 1, 1), 1 / 3) * radius
        t = np.random.uniform(0, np.pi, 1)
        p = np.random.uniform(0, 2 * np.pi, 1)
        x = center[0] + r * np.sin(t) * np.cos(p)
        y = center[1] + r * np.sin(t) * np.sin(p)
        z = center[2] + r * np.cos(t)
    return x, y, z
# 变异
def searchDmutate(population,DF,bestTopPopulation):
    mutated_population = np.empty((len(population) ,len(population[0])),dtype=float)
    for i in range(len(population)):
        ind1 = random.randint(0,len(population) - 1)
        ind2 = random.randint(0,len(population) - 1)
        ind3 = random.randint(0,len(population) - 1)
        while(ind1 == i or ind2 == i or ind3 == i or ind1 == ind2 or ind1 == ind3 or ind2 == ind3 ):
            ind1 = random.randint(0,len(population) - 1)
            ind2 = random.randint(0,len(population) - 1)
            ind3 = random.randint(0,len(population) - 1)
        x = random.randint(0,len(bestTopPopulation) - 1)
        for j in range(len(population[0])):

             mutated_population[i][j] = population[i][j] + DF * (population[ind1][j] -population[ind2][j]) + DF * (bestTopPopulation[x][j]-population[i][j])
    return mutated_population
# 交叉
def searchcross(mutated_population,population,CR,center, radius,params):
    lower = params[0]
    upper = params[1]
    crossed_population = np.empty((len(population),len(population[0])),dtype=float)
    for i in range(len(population)):
        Jrand = random.randint(0,len(population[0]))
        for j in range(len(population[0])):
            if(j == Jrand or random.uniform(0,1) < CR):
                crossed_population[i][j] = mutated_population[i][j]
            else:
                crossed_population[i][j] = population[i][j]
            while crossed_population[i][0] < lower or crossed_population[i][0] > upper or crossed_population[i][1] < lower or crossed_population[i][1] > upper or crossed_population[i][2] < lower or crossed_population[i][2] > upper:

                r = np.power(np.random.uniform(0, 1, 1), 1 / 3) * radius
                t = np.random.uniform(0, np.pi, 1)
                p = np.random.uniform(0, 2 * np.pi, 1)
                crossed_population[i][0]= center[0] + r * np.sin(t) * np.cos(p)
                crossed_population[i][1] = center[1] + r * np.sin(t) * np.sin(p)
                crossed_population[i][2] = center[2] + r * np.cos(t)


    return crossed_population

def searchselect(crossed_population,population,score,func_num,center,radius,params):
    populationd = np.copy(population)
    scored = np.copy(score)
    new_sco = cec2013main(crossed_population,func_num)
    for i in range(len(population)):

        if(new_sco[i] > scored[i]):
            populationd[i] = crossed_population[i]
    for i in range(len(populationd)):

        while is_point_inside_sphere(populationd[i],center,radius) == False :
            populationd[i][0],populationd[i][1],populationd[i][2] = generate_points_in_sphere(center, radius, 1,params)
    selectscore = cec2013main(populationd,func_num)
    return populationd,selectscore




def is_point_inside_sphere(point, center, radius):
    distance = np.sqrt(np.sum((point - center)**2))
    return distance <= radius

def hiv(x,bestPop,func_num):
    tempP = np.linspace(x,bestPop,10)
    tempS = cec2013main(tempP,func_num)
    min = 9999
    for i in range(1,len(tempS)):
        if tempS[i] < min:
            min = tempS[i]
    if min < tempS[0]:
        return False
    else:
        return True


def Search(selectBound,TS,bestPop,bestS,population_size,func_num,cluster_iter,param_num,globalbestScore,params):
    skip_search = []

    selectBoundPOP = np.empty((len(selectBound),param_num))
    selectradious = np.empty((len(selectBound),1))

    selectflag = np.zeros((len(selectBound),1))

    new_TS_index = []
    for i in range(len(selectBound)):
        for j in range(param_num):
            selectBoundPOP[i][j] = selectBound[i][j]
        selectradious[i] = selectBound[i][param_num + 1]
    if len(bestPop) != 0:
        # for i in range(len(selectBoundPOP)):
        #     for j in range(len(bestPop)):
        #         temp = hiv(selectBoundPOP[i],bestPop[j],func_num)
        #         if temp == True:
        #             selectflag[i][0] = -999999
        #             break
        for i in range(len(selectBoundPOP)):
            if Odistance1(TS,selectBoundPOP,i) == 1:
                selectflag[i][0] = -999999

    for cluster in range(len(selectBound)):
        flag = 0
        if selectflag[cluster][0] == -999999:
            # print("当前搜索范围:"+str(selectBound[cluster])+'跳过搜索')
            skip_search.append(cluster)
        else:
            flag = 1
            top = 0.2
            DF = 0.1
            CR = 0.5
            bestPopLen = len(bestPop)

            pop,bestTopPopulation = generate_pop(selectBoundPOP[cluster],selectradious[cluster],population_size,param_num,top,func_num,params)
            # pop,sco = selectBoundInit(selectBound[cluster][0],selectBound[cluster][1],selectBound[cluster][2],population_size,func_num)
            # 精英保留策略
            pop[0][0] = selectBound[cluster][0]
            pop[0][1] = selectBound[cluster][1]
            pop[0][2] = selectBound[cluster][2]
            sco = cec2013main(pop,func_num)
            center = [selectBound[cluster][0], selectBound[cluster][1], selectBound[cluster][2]]
            for it in range(cluster_iter):

                mutated_population = searchDmutate(pop,DF,bestTopPopulation)

                crossed_population = searchcross(mutated_population,pop,CR,selectBoundPOP[cluster],selectradious[cluster],params)

                selectpop,selectsco = searchselect(crossed_population,pop,sco,func_num,center,param_num,params)
                pop = np.copy(selectpop)
                sco = np.copy(selectsco)
                index = desort(selectsco)

                for i in range(len(bestTopPopulation)):
                    bestTopPopulation[i]= selectpop[int(index[i])]
            current_pop = selectpop[int(index[0])]
            current_s = selectsco[int(index[0])]
            globalbestScore = max(globalbestScore,current_s[0])
            if len(bestPop) == 0:
                bestPop= np.empty((1,param_num))
                bestS = np.empty((1,1))
                bestPop[0][0]=current_pop[0]
                bestPop[0][1]=current_pop[1]
                bestPop[0][2] = current_pop[2]
                bestS[0] = current_s[0]
            else:

                bestPop,bestS = panduanbestPop(bestPop,bestS,current_pop,current_s,globalbestScore,param_num)
            if bestPopLen != len(bestPop):
                new_TS_index.append(cluster)
        if flag == 1:
            continue
    #         print("当前搜索范围:"+str(selectBound[cluster])+',' +'最优种群:'+str(current_pop)+',适应度'+str(current_s))
    # print(str(bestPop),str(bestS))

    # new_TS = np.empty((len(new_TS_index),param_num + 1))
    # for i in range(len(new_TS_index)):
    #     for j in range(param_num):
    #         new_TS[i][j]= selectBound[int(new_TS_index[i])][j]
    #     new_TS[i][-1] = selectBound[int(new_TS_index[i])][4]
    # # 合并TS
    # if len(TS) == 0:
    #     TS = new_TS
    # else:
    #     TS = np.concatenate((TS,new_TS),axis = 0)
    # 更新TS
    # TS = updataTS(TS,selectBound,param_num)\
    print('skip_search:'+str(len(skip_search)))
    TS = newupdataTS1(TS,selectBound,param_num,skip_search,new_TS_index)
    print(len(TS))
    return TS,bestPop,bestS,globalbestScore

def panduanbestPop(bestPop,bestS,currentpop,currents,globalbestScore,param_num):
    if abs(currents[0] - globalbestScore) <= 0.001:
        count = 0
        for pop in bestPop:
            if abs(pop[0] - currentpop[0]) < 0.1 and abs(pop[1] - currentpop[1]) < 0.1 and abs(pop[2] - currentpop[2]) < 0.1:
                count += 1
        if count == 0:
            bestPop1 = np.empty((1, param_num))
            bestS1 = np.empty((1, 1))
            bestPop1[0][0] = currentpop[0]
            bestPop1[0][1] = currentpop[1]
            bestPop1[0][2] = currentpop[2]
            bestS1[0] = currents[0]
            bestPop = np.concatenate((bestPop, bestPop1), axis=0)
            bestS = np.concatenate((bestS, bestS1), axis=0)
    return bestPop,bestS

def updataTS(TS,ParamPeak,dim):
    new_TS = np.empty((len(ParamPeak), dim + 1))
    for i in range(len(ParamPeak)):
        for j in range(dim):
            new_TS[i][j] = ParamPeak[i][j]
        new_TS[i][-1] = ParamPeak[i][dim+1]
    TS = np.concatenate((TS, new_TS), axis=0)
    return TS
def newupdataTS(TS,ParamPeak,dim,up_TS_index):
    updateTS = np.empty((len(up_TS_index),dim + 1))
    count = -1
    for index in up_TS_index:
        count += 1
        for j in range(dim):
            updateTS[count][j] = ParamPeak[index][j]
        updateTS[count][-1] = ParamPeak[index][dim+1]
    TS = np.concatenate((TS, updateTS), axis=0)
    return TS
def newupdataTS1(TS,ParamPeak,dim,up_TS_index,new_TS_index):
    new_TS = np.empty((len(new_TS_index),dim + 1))
    count = -1
    for index in new_TS_index:
        count += 1
        for j in range(dim):
            new_TS[count][j] = ParamPeak[index][j]
        new_TS[count][-1] = ParamPeak[index][dim + 1]
    TS = np.concatenate((TS, new_TS), axis=0)
    for t in TS:
        for up in up_TS_index:
            sum = 0
            for i in range(dim):
                 sum += (t[i] - ParamPeak[up][i]) ** 2
            sum = math.sqrt(sum)
            if sum < t[-1]:
                t[-1] = min(max(ParamPeak[up][dim+1],t[-1]),1)



    return TS

def generate_pop(center, radius, population_size,dim,top,func_num,params):
    lower = params[0]
    upper = params[1]

    radii = np.power(np.random.uniform(0, 1, population_size), 1 / 3) * radius
    theta = np.random.uniform(0, np.pi, population_size)
    phi = np.random.uniform(0, 2 * np.pi, population_size)

    x = center[0] + radii * np.sin(theta) * np.cos(phi)
    y = center[1] + radii * np.sin(theta) * np.sin(phi)
    z = center[2] + radii * np.cos(theta)
    for i in range(population_size):
        while z[i] < lower or z[i] > upper or x[i] < lower or x[i] > upper or y[i] < lower or y[i] > upper:
            r = np.power(np.random.uniform(0, 1, 1), 1 / 3) * radius
            t = np.random.uniform(0, np.pi, 1)
            p = np.random.uniform(0, 2 * np.pi, 1)
            x[i] = center[0] + r * np.sin(t) * np.cos(p)
            y[i] = center[1] + r * np.sin(t) * np.sin(p)
            z[i] = center[2] + r * np.cos(t)
    pop = np.vstack((np.vstack((x, y)), z)).T
    return pop,pop[:int(top * population_size)]

def partation(x_lower,x_upper):

    x_splits = np.linspace(x_lower, x_lower + (x_upper - x_lower) / 2, num=2)
    y_splits = np.linspace(x_lower, x_lower + (x_upper - x_lower) / 2, num=2)
    z_splits = np.linspace(x_lower, x_lower + (x_upper - x_lower) / 2, num=2)


    sections = []
    for x_start in x_splits:
        for y_start in y_splits:
            for z_start in z_splits:
                x_end = x_start + (x_splits[1] - x_splits[0])
                y_end = y_start + (y_splits[1] - y_splits[0])
                z_end = z_start + (z_splits[1] - z_splits[0])
                section = [[x_start, y_start, z_start],
                           [x_end, y_start, z_start],
                           [x_start, y_end, z_start],
                           [x_end, y_end, z_start],
                           [x_start, y_start, z_end],
                           [x_end, y_start, z_end],
                           [x_start, y_end, z_end],
                           [x_end, y_end, z_end]]
                sections.append(section)
    return sections