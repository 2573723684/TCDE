import random
import numpy as np
import AloUtils
pi = 3.1415926535897932384626433832795029
def run(func_num,population_size,top,dim,param_bound,DF,DCR,iter_num,EF,ECR,TS,max_population,cluster_count):
    # 初始化最优种群
    # 初始化种群，均匀分布
    population = np.random.uniform(param_bound[0], param_bound[1], size=(population_size, dim))
    scores = AloUtils.cec2013main(population, func_num)
    bestTopPopulation = np.random.uniform(param_bound[0], param_bound[1],size=(int(population_size * top), dim))
    initpop = np.copy(population)
    inits = np.copy(scores)
    for iter in range(iter_num):
        # 开发种群交叉，变异，选择
        develop_mutated_population = Dmutate(population,DF,bestTopPopulation)
        develop_crossed_population = cross(develop_mutated_population,population,DCR,param_bound,dim)
        develop_selected_population,develop_selected_score= select(develop_crossed_population,population,scores,func_num,TS,param_bound,dim)
        # 探索种群交叉，变异，选择
        explore_mutated_population = Emutate(population,EF)
        explore_crossed_population =cross(explore_mutated_population,population,ECR,param_bound,dim)
        explore_selected_population,explore_selected_score = select(explore_crossed_population, population, scores,func_num,TS,param_bound,dim)
        # 开发种群与探索种群合并
        merged_population,merged_population_scores,bestTopPopulation,current_best,current_best_scores= merge(develop_selected_population,explore_selected_population,develop_selected_score,explore_selected_score,func_num,top)
        population = np.copy(merged_population)
        scores = np.copy(merged_population_scores)
        # 保存所有的个体
        if iter == 0:
            all_population,all_scores = AloUtils.savePopulation_Scores(population,initpop,scores,inits)
        if iter >= 0:
            all_population, all_scores = AloUtils.savePopulation_Scores(all_population,population,all_scores,scores)
        if len(all_population) + cluster_count >= max_population or iter == 100:
            print('iter:' + str(iter))
            print('max_population:' + str(len(all_population)))
            return all_population,all_scores,iter,current_best,current_best_scores

def select(crossed_population,population,score,func_num,TS,param_bound,dim):
    populationd = np.copy(population)
    scored = np.copy(score)
    new_sco = AloUtils.cec2013main(crossed_population,func_num)
    for i in range(len(population)):
        #最大值问题用 >，设置一个概率，增加其多样性
        if(new_sco[i] > scored[i]):
            populationd[i] = crossed_population[i]
            scored[i] = new_sco[i]
    for i in range(len(populationd)):
        # 如果在禁忌区域，则随机出去
        Ts_count1 = 1
        Ts_count1 = Ts_count1 - AloUtils.Odistance(TS, populationd, i)
        if Ts_count1 != 0:
            while Ts_count1 != 0:
                # 如果在禁忌区域，则随机出去
                Ts_count1 = 1
                populationd[i] = np.random.uniform(param_bound[0], param_bound[1], size=(1, dim))
                Ts_count1 = Ts_count1 - AloUtils.Odistance(TS, populationd, i)
    scored = AloUtils.cec2013main(populationd,func_num)
    return populationd,scored


def Dmutate(population,DF,besttop):
    mutated_population = np.empty((len(population),len(population[0])),dtype=float)
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

def cross(mutated_population,population,CR,param_bound,dim):
    crossed_population = np.empty((len(population), len(population[0])), dtype=float)
    for i in range(len(population)):
        Jrand = np.random.randint(0, len(population[0]))
        for j in range(len(population[0])):
            if (j == Jrand or np.random.uniform(0, 1) < CR):
                crossed_population[i][j] = mutated_population[i][j]
            else:
                crossed_population[i][j] = population[i][j]
        for j in range(len(population[0])):
            if crossed_population[i][j] > param_bound[1] or crossed_population[i][j] < param_bound[0]:
                crossed_population[i] = np.random.uniform(param_bound[0], param_bound[1], size=(1, dim))
                break
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


