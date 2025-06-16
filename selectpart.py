import numpy as np
import DifferentialEvolution4SeachParam
import AloUtils
import MinDE
def selectMinPart(param_bound,func_num,rad):
    population_size = 200
    k_count = 1
    iter_num = 2000
    DevelopF = 0.9
    DevelopCR = 0.1
    ExploreF = 0.5
    ExploreCR = 0.9
    param_num = 2
    top = 0.1
    radious = 0.1
    selectParam = 0.7
    selectParamStart = 0
    print("开始selectMinPart")
    all_part_pop,all_part_score = DifferentialEvolution4SeachParam.run(func_num,population_size,top,param_num,param_bound,DevelopF,DevelopCR,iter_num,ExploreF,ExploreCR)
    MulimodalPopulation, range_step, MulimodalScores = AloUtils.Min(all_part_pop, func_num, all_part_score, k_count, param_bound)
    selectMountPop, selectMountS = AloUtils.dataSelect(MulimodalPopulation, range_step,MulimodalScores, param_bound,func_num, selectParam,selectParamStart,999)
    clusters = AloUtils.dbscn(selectMountPop, func_num, param_bound,rad)
    bestPopArray = MinDE.Minde(selectMountPop,selectMountS,clusters,func_num,radious,param_bound,iter_num)
    bestPopArray = np.array(list(set([tuple(t) for t in bestPopArray])))
    print('bestPopArray:'+str(bestPopArray))
    return bestPopArray

