import DifferentialEvolution4SeachParam
import random
import numpy as np
import AloUtils
import MinDE
def selectDistant(population_size,param_bound,func_num,DE_iter,TS,allbestPop):
    DF = 0.5
    DCR = 0.1
    EF = 0.5
    ECR = 0.9
    bestPopArray,current_pop = selectDistantDE(population_size,param_bound,func_num,DE_iter,TS,0.1,2,DF,DCR,EF,ECR,allbestPop)
    return bestPopArray,current_pop
def selectDistantDE(population_size,param_bound,func_num,DE_iter,TS,top,param_num,DF,DCR,EF,ECR,allbestPop):
    bestTopPopulation = np.empty([int(population_size * top), param_num], dtype=float)
    select_dist_population,select_dist_scores = initTSPopulation(population_size,TS,param_bound,func_num,)
    initpop = np.copy(select_dist_population)
    inits = np.copy(select_dist_scores)
    lastBest = [-999,-999,-999999]
    for iter in range(DE_iter):
        develop_mutated_population = Dmutate(select_dist_population, DF, bestTopPopulation)
        develop_crossed_population = cross(develop_mutated_population, select_dist_population, DCR, param_bound,TS)
        develop_selected_population, develop_selected_score = select(develop_crossed_population, select_dist_population, select_dist_scores,func_num)
        explore_mutated_population = Emutate(select_dist_population, EF)
        explore_crossed_population = cross(explore_mutated_population, select_dist_population, ECR, param_bound,TS)
        explore_selected_population, explore_selected_score = select(explore_crossed_population, select_dist_population, select_dist_scores,
                                                                     func_num)
        merged_population,merged_population_scores,bestTopPopulation,current_best,current_best_scores= merge(develop_selected_population,explore_selected_population,develop_selected_score,explore_selected_score,func_num,top)
        select_dist_population = np.copy(merged_population)
        select_dist_scores = np.copy(merged_population_scores)
        print("selectDist:" + str(current_best) + "/" + str(current_best_scores))
        if iter % 400 == 0:
            print(str(current_best)+'/'+str(current_best_scores))
            if lastBest[0] == current_best[0] and lastBest[1] == current_best[1] and lastBest[2] == current_best_scores:
                break
            else:
                lastBest = [current_best[0], current_best[1], current_best_scores]
    current_pop, current_s = SortPop(select_dist_population, select_dist_scores)
    # print("selectDist:"+str(current_s) + "/" + str(current_pop))
    bestpop = np.empty((1, 2))
    bestpop[0][0] = allbestPop[0][0]
    bestpop[0][1] = allbestPop[0][1]
    bestS = AloUtils.cec2013main(allbestPop,func_num)
    bestScore = bestS[0]
    bestPopArray = selectAllPop(select_dist_population, select_dist_scores, bestpop, bestScore, param_num)
    current_pop1 = np.empty((1,2))
    current_pop1[0][0] = current_pop[0]
    current_pop1[0][1] = current_pop[1]
    return bestPopArray,current_pop1

def select(crossed_population,population,score,func_num):
    populationd = np.copy(population)
    scored = np.copy(score)
    new_sco = AloUtils.cec2013main(crossed_population,func_num)
    for i in range(len(population)):
        #最大值问题用 >
        if(new_sco[i] > scored[i]):
            populationd[i] = crossed_population[i]
            scored[i] = new_sco[i]
    return populationd,scored

def Dmutate(population,DF,besttop):
    mutated_population = np.empty((len(population) ,len(population[0])),dtype=float)
    for i in range(len(population)):
        ind1 = random.randint(0,len(population) - 1)
        ind2 = random.randint(0,len(population) - 1)
        ind3 = random.randint(0,len(population) - 1)
        while(ind1 == i or ind2 == i or ind3 == i or ind1 == ind2 or ind1 == ind3 or ind2 == ind3 ):
            ind1 = random.randint(0,len(population) - 1)
            ind2 = random.randint(0,len(population) - 1)
            ind3 = random.randint(0,len(population) - 1)
        x = random.randint(0,len(besttop) - 1)
        for j in range(len(population[0])):
             mutated_population[i][j] = population[i][j] + DF * (population[ind1][j] - population[ind2][j]) + DF * (besttop[x][j]-population[i][j])
    return mutated_population

def Emutate(population,EF):
    mutated_population = np.empty((len(population),len(population[0])), dtype=float)
    for i in range(len(population)):
        ind1 = random.randint(0, len(population) - 1)
        ind2 = random.randint(0, len(population) - 1)
        ind3 = random.randint(0, len(population) - 1)
        while (ind1 == i or ind2 == i or ind3 == i or ind1 == ind2 or ind1 == ind3 or ind2 == ind3):
            ind1 = random.randint(0, len(population) - 1)
            ind2 = random.randint(0, len(population) - 1)
            ind3 = random.randint(0, len(population) - 1)
        x = random.uniform(0,1)
        for j in range(len(population[0])):
            mutated_population[i][j] = population[i][j] + EF * (population[ind1][j] -population[ind2][j]) + x * (population[ind3][j]-population[i][j])
    return mutated_population

def cross(mutated_population,population,CR,param_bound,TS):
    Ts = TS.T
    crossed_population = np.empty((len(population),len(population[0])),dtype=float)
    for i in range(len(population)):
        Jrand = random.randint(0,len(population[0]))
        for j in range(len(population[0])):
            if(j == Jrand or random.uniform(0,1) < CR):
                crossed_population[i][j] = mutated_population[i][j]
            else:
                crossed_population[i][j] = population[i][j]

        Ts_count1 = 0
        if crossed_population[i][0] >= param_bound[0] and crossed_population[i][0] <= param_bound[1] and crossed_population[i][1] >= param_bound[2] and crossed_population[i][1] <= param_bound[3]:
            Ts_count1 += 1

        for t in range(len(Ts)):
            if (Ts[t][0] > crossed_population[i][0] or Ts[t][1] < crossed_population[i][0]) or (Ts[t][2] > crossed_population[i][1] or Ts[t][3] < crossed_population[i][1]):
                Ts_count1 += 1
        if Ts_count1 != len(Ts) + 1:
            while Ts_count1 != len(Ts) + 1:
                Ts_count1 = 0
                crossed_population[i][0] = random.uniform(param_bound[0],param_bound[1])
                crossed_population[i][1] = random.uniform(param_bound[2],param_bound[3])
                if crossed_population[i][0] > param_bound[0] and crossed_population[i][0] < param_bound[1] and crossed_population[i][1] > param_bound[2] and crossed_population[i][1] < param_bound[3]:
                    Ts_count1 += 1
                for t in range(len(Ts)):
                    if (Ts[t][0] > crossed_population[i][0] or Ts[t][1] < crossed_population[i][0]) or (
                            Ts[t][2] > crossed_population[i][1] or Ts[t][3] < crossed_population[i][1]):
                        Ts_count1 += 1

    return crossed_population

def merge(develop_population,explore_population,develop_selected_score,explore_selected_score,func_num,top):
    merge_population = np.empty((len(develop_population),len(develop_population[0])),dtype=float)
    develop_sort_population_index = AloUtils.desort(develop_selected_score)
    for i in range(int(len(develop_population) * 0.5)):
        temp = int(develop_sort_population_index[i])
        for j in range(len(develop_population[0])):
            merge_population[i][j] = develop_population[temp][j]
    explore_sort_population_index = AloUtils.desort(explore_selected_score)
    for i in range(int(len(explore_population)*0.2),len(explore_population)):
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

def initTSPopulation(population_size,TS,param_bound,func_num):
    ts = TS.T
    select_dist_population = np.empty((population_size,2),dtype=float)
    for i in range(population_size):
        TS_count1 = 0
        while TS_count1 != len(ts):
            TS_count1 = 0
            select_dist_population[i][0] = random.uniform(param_bound[0], param_bound[1])
            select_dist_population[i][1] = random.uniform(param_bound[2], param_bound[3])
            for t in range(len(ts)):
                if (ts[t][0] > select_dist_population[i][0] or ts[t][1] < select_dist_population[i][0]) or (
                        ts[t][2] > select_dist_population[i][1] or ts[t][3] < select_dist_population[i][1]):
                    TS_count1 += 1
    select_dist_scores = AloUtils.cec2013main(select_dist_population,func_num)
    return select_dist_population,select_dist_scores

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

if __name__ == '__main__':
    f12 = [-5, 5, -5, 5, 8, 12]
    param_bound = f12
    allbestPop = np.empty((1,2))
    allbestPop[0][0] = 1.757743
    allbestPop[0][1] = 1.595737
    TS = np.array([[-4,-3,-4,-3],[0,1,3,4],[1,2,1,2],[-2,-1,4,5],[-2,-1,3,4],[-3,-2,1,2],[-1,0,-5,-4]])
    b = selectDistant(200,param_bound,param_bound[5],2000,TS,allbestPop)
    #4.141256932056421, 2.4770118064717774

    # [[-3.39511302 - 3.3173072]
    #  [0.16287607  3.78150997]
    # [1.75774359
    # 1.59573726]
    # [-1.5615447   4.40002067]
    # [-1.48147154
    # 3.67402269]
    # [-2.18498435  1.68705394]
    # [-0.49969507 - 4.01259708]]
