import numpy as np
import AloUtils
import random
import DifferentialEvolution4SeachParam
def jugPeak(population,scores,clusters,param_bound,func_num,population_size):
    cluster_pop = clustersPeak(clusters,population,scores)
    cluster_s = AloUtils.cec2013main(cluster_pop,param_bound[5])
    peakType = judgePeakType(cluster_pop,cluster_s,param_bound)
    Zone1_bound,Zone2_bound,Zone3_bound,Zone4_bound = scopeDivision4(param_bound)
    zoneType = peakRegion(cluster_pop,peakType,Zone1_bound,Zone2_bound,Zone3_bound,Zone4_bound)
    for i in range(len(zoneType)):
        top = 0.1
        param_num = 2
        iter_num = 2000
        if i == 0:
            pparam_bound = Zone1_bound
            DF, EF, DCR, ECR, Dmerge, EMerge = getpram(zoneType[i][0],zoneType[i][1])
            search(population_size,top,param_num,pparam_bound,func_num,iter_num,DF,EF,DCR,ECR,Dmerge,EMerge)


def search(population_size,top,param_num,param_bound,func_num,iter_num,DF,EF,DCR,ECR,Dmerge,EMerge):
    bestTopPopulation = np.empty([int(population_size * top), param_num], dtype=float)
    population, scores = DifferentialEvolution4SeachParam.init(population_size, param_num, param_bound, func_num)
    for iter in range(iter_num):

        develop_mutated_population = DifferentialEvolution4SeachParam.Dmutate(population,DF,bestTopPopulation)
        develop_crossed_population = DifferentialEvolution4SeachParam.cross(develop_mutated_population,population,DCR,param_bound)
        develop_selected_population,develop_selected_score= DifferentialEvolution4SeachParam.select(develop_crossed_population,population,scores,func_num)
        explore_mutated_population = DifferentialEvolution4SeachParam.Emutate(population,EF)
        explore_crossed_population =DifferentialEvolution4SeachParam.cross(explore_mutated_population,population,ECR,param_bound)
        explore_selected_population,explore_selected_score = DifferentialEvolution4SeachParam.select(explore_crossed_population, population, scores,func_num)
        merged_population, merged_population_scores, bestTopPopulation, current_best, current_best_scores = merge(
            develop_selected_population, explore_selected_population, develop_selected_score, explore_selected_score,
            func_num, top,Dmerge,EMerge)
        population = np.copy(merged_population)
        scores = np.copy(merged_population_scores)
        print(current_best_scores)
        if iter % 100 == 0 and EMerge != 1:
            Dmerge += 0.1
            EMerge -= 0.1
            DF -= 0.1
            EF -= 0.1
        if DF <= 0.1:
            DF = 0.1
        if EF <=0.1:
            EF = 0.1

def getpram(x,y):
    if x >= y:
        DF = 0.5
        EF = 0.5
        DCR = 0.9
        ECR = 0.9
        Dmerge = 0.5
        EMerge = 0.5
    else:
        DF = 0.9
        EF = 0.9
        DCR = 0.9
        ECR = 0.9
        Dmerge = 0.0
        EMerge = 1
    return DF,EF,DCR,ECR,Dmerge,EMerge

def merge(develop_population,explore_population,develop_selected_score,explore_selected_score,func_num,top,Dmerge,EMerge):
    merge_population = np.empty((len(develop_population),len(develop_population[0])),dtype=float)
    develop_sort_population_index = AloUtils.desort(develop_selected_score)
    for i in range(int(len(develop_population) * Dmerge)):
        temp = int(develop_sort_population_index[i])
        for j in range(len(develop_population[0])):
            merge_population[i][j] = develop_population[temp][j]
    explore_sort_population_index = AloUtils.desort(explore_selected_score)
    for i in range(int(len(explore_population)* EMerge),len(explore_population)):
        for j in range(len(explore_population[0])):
            merge_population[i][j] = explore_population[int(explore_sort_population_index[i])][j]


    bestTopPopulation = np.empty((int(len(develop_population)*top),len(develop_population[0])),dtype=float)
    bestTopPopulationScores = np.empty((int(len(develop_population)*top),1),dtype=float)
    merge_population_score = AloUtils.cec2013main(merge_population,func_num)
    merge_sort_population_index = AloUtils.desort(merge_population_score)
    merge_sort_population = np.empty((len(develop_population),len(develop_population[0])),dtype=float)

    for i in range(len(develop_population)):
        for j in range(len(develop_population[0])):
            merge_sort_population[i][j] = merge_population[int(merge_sort_population_index[i])][j]

    for i in range(int(len(develop_population) * top)):
        bestTopPopulationScores[i] = merge_population_score[int(merge_sort_population_index[i])]
        for j in range(len(develop_population[0])):
            bestTopPopulation[i][j] = merge_sort_population[i][j]

    merge_sort_population_scores = AloUtils.cec2013main(merge_sort_population, func_num)
    current_best = merge_sort_population[0]
    current_best_scores = merge_sort_population_scores[0]
    return merge_sort_population,merge_sort_population_scores,bestTopPopulation,current_best,current_best_scores


def peakRegion(cluster_pop,peakType,Zone1_bound,Zone2_bound,Zone3_bound,Zone4_bound):
    Zone1 = []
    Zone2 = []
    Zone3 = []
    Zone4 = []
    for i in range(len(cluster_pop)):
        if cluster_pop[i][0] > Zone1_bound[0] and cluster_pop[i][0] < Zone1_bound[1] and cluster_pop[i][1] > Zone1_bound[2] and cluster_pop[i][1] < Zone1_bound[3]:
            Zone1.append(peakType[i])
        elif cluster_pop[i][0] > Zone2_bound[0] and cluster_pop[i][0] < Zone2_bound[1] and cluster_pop[i][1] > Zone2_bound[2] and cluster_pop[i][1] < Zone2_bound[3]:
            Zone2.append(peakType[i])
        elif cluster_pop[i][0] > Zone3_bound[0] and cluster_pop[i][0] < Zone3_bound[1] and cluster_pop[i][1] > Zone3_bound[2] and cluster_pop[i][1] < Zone3_bound[3]:
            Zone3.append(peakType[i])
        elif cluster_pop[i][0] > Zone4_bound[0] and cluster_pop[i][0] < Zone4_bound[1] and cluster_pop[i][1] > Zone4_bound[2] and cluster_pop[i][1] < Zone4_bound[3]:
            Zone4.append(peakType[i])
        else:
            print("peakRegion have some problem")
    zoneType = np.empty((4,2))

    coarsePeak = 0
    finePpeak = 0
    for i in range(len(Zone1)):
        if Zone1[i] == 0:
            coarsePeak += 1
        else:
            finePpeak += 1
    zoneType[0][0] = coarsePeak
    zoneType[0][1] = finePpeak

    coarsePeak = 0
    finePpeak = 0

    for i in range(len(Zone2)):
        if Zone2[i] == 0:
            coarsePeak += 1
        else:
            finePpeak += 1
    zoneType[1][0] = coarsePeak
    zoneType[1][1] = finePpeak

    coarsePeak = 0
    finePpeak = 0
    for i in range(len(Zone3)):
        if Zone3[i] == 0:
            coarsePeak += 1
        else:
            finePpeak += 1
    zoneType[2][0] = coarsePeak
    zoneType[2][1] = finePpeak

    coarsePeak = 0
    finePpeak = 0
    for i in range(len(Zone4)):
        if Zone4[i] == 0:
            coarsePeak += 1
        else:
            finePpeak += 1
    zoneType[3][0] = coarsePeak
    zoneType[3][1] = finePpeak
    return zoneType



def getNonRepeatList2(data):
    new_data = []
    for i in range(len(data)):
        if data[i] not in new_data:
            new_data.append(data[i])
    return new_data
def clustersPeak(clusters,population,scores):
    reback_pop = []
    reback = []
    new_data = getNonRepeatList2(clusters)
    temp = np.empty((len(new_data), 2))
    for i in range(len(new_data)):
        temp[i][0] = new_data[i]
        temp[i][1] = 0
    for i in range(len(clusters)):
        for j in range(len(temp)):
            if temp[j][0] == clusters[i]:
                temp[j][1] += 1
    for i in range(len(temp)):
        reback.append(temp[i][0])
    for i in range(len(reback)):
        reback_pos = [-100,-100,-9999999]
        for j in range(len(clusters)):
            if reback[i] == clusters[j]:
                if scores[j] > reback_pos[2]:
                    reback_pos[0] = population[j][0]
                    reback_pos[1] = population[j][1]
                    reback_pos[2] = scores[j]
        reback_pop.append(reback_pos[0])
        reback_pop.append(reback_pos[1])
    cluster_pop = np.empty(int((len(reback_pop) / 2)),2)
    count = 0
    for i in range(len(cluster_pop)):
        cluster_pop[i][0] = reback_pop[count]
        count += 1
        cluster_pop[i][1] = reback_pop[count]
        count += 1
    return cluster_pop

def judgePeakType(cluster_pop,cluster_s,param_bound):
    maximumGap = np.empty((len(cluster_pop),1))
    for i in range(cluster_pop):
        tempx = np.random.uniform(cluster_pop[i][0] - 0.1,cluster_pop[i][0] + 0.1,(10,1))
        tempy = np.random.uniform(cluster_pop[i][1] - 0.1,cluster_pop[i][1] + 0.1,(10,1))
        tempz = np.concatenate((tempx, tempy), axis=1)
        f = AloUtils.cec2013main(tempz,param_bound[5])
        max = 0
        for j in range(len(f)):
            if abs(cluster_s[i] - f[j]) > maximumGap[i]:
                maximumGap[i] = abs(cluster_s[i] - f[j])
                if maximumGap[i] > max:
                    max = maximumGap[i]
    #判断适应度变化程度来判断峰的类型
    for i in range(len(maximumGap)):
        if maximumGap[i] <= 10:
            maximumGap[i] = 0#粗峰
        else:
            maximumGap[i] = 1#细峰
    return maximumGap

def scopeDivision4(param_bound):
    Zone1_bound = [param_bound[0], param_bound[0] + (param_bound[1] - param_bound[0]) / 2, param_bound[2],
                   param_bound[2] + (param_bound[3] - param_bound[2]) / 2]
    Zone2_bound = [param_bound[0] + (param_bound[1] - param_bound[0]) / 2, param_bound[1], param_bound[2],
                   param_bound[2] + (param_bound[3] - param_bound[2]) / 2]
    Zone3_bound = [param_bound[0], param_bound[0] + (param_bound[1] - param_bound[0]) / 2,
                   param_bound[2] + (param_bound[3] - param_bound[2]) / 2, param_bound[3]]
    Zone4_bound = [param_bound[0] + (param_bound[1] - param_bound[0]) / 2, param_bound[1],
                   param_bound[2] + (param_bound[3] - param_bound[2]) / 2, param_bound[3]]
    return Zone1_bound,Zone2_bound,Zone3_bound,Zone4_bound