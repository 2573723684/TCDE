import numpy as np
import DifferentialEvolution4SeachParam
import AloUtils
def Minde(population,scores,clusters,func_num,radious,param_bound,DE_iter):

    temp = np.empty((len(population), 4))
    for i in range(len(population)):
        for j in range(len(population[0])):
            temp[i][j] = population[i][j]
        temp[i][2] = scores[i]
        temp[i][3] = clusters[i]
    Multiple_nodes_num = max(clusters) + 1
    for Multiple_node in range(Multiple_nodes_num):
        parameter = [-100, -100, -99999999, radious]
        for i in range(len(temp)):
            if Multiple_node == temp[i][3] and temp[i][2] >= parameter[2]:
                parameter[0] = temp[i][0]
                parameter[1] = temp[i][1]
                parameter[2] = temp[i][2]

        if Multiple_node == 0:

            bestPopArray = optimalSolution(parameter, func_num, param_bound, DE_iter)
        else:
            bestPopArray1 = optimalSolution(parameter, func_num, param_bound, DE_iter)
            bestPopArray = savePopulation_Scores(bestPopArray1,bestPopArray)

    return bestPopArray

def optimalSolution(paramter, func_num, param_bound, DE_iter):
    Range = [paramter[0] - paramter[3] * 3, paramter[0] + paramter[3] * 3, paramter[1] - paramter[3] * 3,
             paramter[1] + paramter[3] * 3]
    if Range[0] < param_bound[0]:
        Range[0] = param_bound[0]
    if Range[1] > param_bound[1]:
        Range[1] = param_bound[1]
    if Range[2] < param_bound[2]:
        Range[2] = param_bound[2]
    if Range[3] > param_bound[3]:
        Range[3] = param_bound[3]
    population_size = 200
    param_num = 2
    top = 0.1
    DF = 0.9
    DCR = 0.1
    EF = 0.5
    ECR = 0.9
    min_population, min_scores = DifferentialEvolution4SeachParam.init(population_size,param_num,Range,func_num)
    bestTopPopulation = np.empty([int(population_size * top), param_num], dtype=float)
    lastBest = [999,999,-99999]
    for i in range(DE_iter):
        develop_mutated_population = DifferentialEvolution4SeachParam.Dmutate(min_population, DF, bestTopPopulation)
        develop_crossed_population = DifferentialEvolution4SeachParam.cross(develop_mutated_population, min_population, DCR, param_bound)
        develop_selected_population, develop_selected_score = DifferentialEvolution4SeachParam.select(develop_crossed_population, min_population, min_scores,
                                                                     func_num)
        explore_mutated_population = DifferentialEvolution4SeachParam.Emutate(min_population, EF)
        explore_crossed_population = DifferentialEvolution4SeachParam.cross(explore_mutated_population, min_population, ECR, param_bound)
        explore_selected_population, explore_selected_score = DifferentialEvolution4SeachParam.select(explore_crossed_population, min_population, min_scores,
                                                                     func_num)
        merged_population, merged_population_scores, bestTopPopulation, current_best, current_best_scores = DifferentialEvolution4SeachParam.merge(
            develop_selected_population, explore_selected_population, develop_selected_score, explore_selected_score,
            func_num, top)
        min_population = np.copy(merged_population)
        min_scores = np.copy(merged_population_scores)
        if i % 50 == 0:
            if lastBest[0] == current_best[0] and lastBest[1] == current_best[1] and lastBest[2] == current_best_scores:
                break
            else:
                lastBest = [current_best[0],current_best[1],current_best_scores]


    current_pop, current_s = SortPop(min_population, min_scores)
    print(str(current_s) +"/"+ str(current_pop))
    pop = np.empty((1, 2))
    pop[0][0] = current_pop[0]
    pop[0][1] = current_pop[1]
    bestPopArray = selectAllPop(min_population, min_scores, pop, current_s, param_num)

    return bestPopArray
import AloUtils
def SortPop(pop,scores):
    sort_population_index = AloUtils.desort(scores)
    sorted_pop = np.empty((len(pop),len(pop[0])))
    sorted_scores = np.empty((len(scores),1))
    for i in range(len(pop)):
        for j in range(len(pop[0])):
            sorted_pop[i][j] = pop[int(sort_population_index[i])][j]
        sorted_scores[i] = scores[int(sort_population_index[i])]
    return sorted_pop[0],sorted_scores[0]

def selectAllPop(population, scores, currentBestPop, currentBestS,param_num):

    for i in range(len(population)):
        if abs(currentBestS - scores[i]) <= 0.00001:
            counts = 0
            for j in range(len(currentBestPop)):
                if abs(population[i][0] - currentBestPop[j][0]) <= 0.5 or abs(population[i][1] - currentBestPop[j][1]) <= 0.5 :
                    counts += 1
            if counts == 0:
                x = np.empty((1,param_num))
                x[0][0] = population[i][0]
                x[0][1] = population[i][1]
                currentBestPop = savePopulation_Scores(currentBestPop,x)

    return currentBestPop

def savePopulation_Scores(all_population, current_population):
    x = []
    num = len(current_population)
    for i in range(len(current_population)):
        for j in range(len(all_population)):
            if all_population[j][0] == current_population[i][0] and all_population[j][1] == current_population[i][1]:
                num = num - 1
                x.append(i)
                break

    tempPopulatiaon = np.empty((num + len(all_population),len(all_population[0])), dtype=float)

    for i in range(len(all_population)):
        tempPopulatiaon[i][0] = all_population[i][0]
        tempPopulatiaon[i][1] = all_population[i][1]

    count = len(all_population)
    for i in range(len(current_population)):
        if i not in x:
            tempPopulatiaon[count][0] = current_population[i][0]
            tempPopulatiaon[count][1] = current_population[i][1]
            count = count + 1
    return np.copy(tempPopulatiaon)