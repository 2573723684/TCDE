from DifferentialEvolution4SeachParam import *
from AloUtils import *
import ada
dim = 3
population_size = 200
iter_num = 50000
DevelopF = 0.5
DevelopCR = 0.1
ExploreF = 0.5
ExploreCR = 0.9
top = 0.1
f4 = [-6,6,-6,6,4,4]
f5 = [-1.9,1.9,-1.9,1.9,2,5]
f6 = [-10,10,-10,10,18,6]
f7 = [0.25,10,0.25,10,36,7]
f8 = [-10,10,-10,10,81,8]
f9 = [0.25,10,0.25,10,216,9]
f10 = [0,1,0,1,12,10]
f11 = [-5,5,-5,5,6,11]
f12 = [-5,5,-5,5,8,12]
f13 = [-5,5,-5,5,6,13]
f14 = [-5,5,-5,5,6,14]
f15 = [-5,5,-5,5,8,15]
N = 0.4
#         4: himmelblau,
#         5: six_hump_camel_back,
#         6: shubert,
#         7: vincent,
#         8: 3Dshubert, 81
#         9: 3Dvincent, 216
#         10: modified_rastrigin_all,
#         11: CF1, 6
#         12: CF2, 6
#         13: CF3, 6
#         14: 3DCF3, 6
#         15: CF4,  8

if __name__ == '__main__':
    maxIter = 100000
    NRs = 50
    MaxFEs = iter_num
    NPFs = 0
    PR = 0
    NSR = 0
    FES = 0
    param_bound = f9
    NP = param_bound[5]
    func_num = param_bound[5]
    selectParam = 0.2
    radious = (param_bound[1] - param_bound[0]) / 200
    min_points = 3
    max_population = 1500
    for NR in range(50):
        max_population = 2000
        countiter = 0
        cluster_count = 0
        globalbestScore = -999999
        TS = np.empty((0,dim + 1))
        all_population = np.empty((0,dim))
        selectAllPOP = np.empty((0,dim))
        selectAllS = np.empty((0,1))
        bestPop = np.empty((0,dim))
        bestS = np.empty((0,1))
        for j in range(50):

            error = 0
            all_population, all_scores,countiter,current_best,current_best_scores= run(func_num,population_size,top,dim,param_bound,DevelopF,DevelopCR,iter_num,ExploreF,ExploreCR,TS,max_population,cluster_count)

            import paints
            paints.paint(all_population, param_bound[0], param_bound[1], func_num, j)
            MulimodalPopulation,  MulimodalScores = Min(all_population, func_num, all_scores, param_bound)

            if error == 0:
                selectMountPop, selectMountS = dataSelect(MulimodalPopulation, MulimodalScores, param_bound,param_bound[5], selectParam,0,NR)

                if j == 0:
                    selectMountPop = np.concatenate((selectMountPop, selectAllPOP), axis=0)
                    selectMountS = np.concatenate((selectMountS, selectAllS), axis=0)
                if j >= 0:
                    selectMountPop,selectMountS = AloUtils.savePopulation_Scores(selectAllPOP, selectMountPop, selectAllS,selectMountS)
                selectAllPOP = selectMountPop
                selectAllS = selectMountS

                clusters,clusterPop,clusterS= dbscn(selectMountPop,radious,min_points,selectMountS)

                selectBound = baseClusterSearch(clusters,selectMountPop,selectMountS,bestPop)
                # ada.judgeByRegion(selectBound,bestPop,param_bound)

                cluster_iter = 100

                TS,bestPop,bestS,globalbestScore = Search(selectBound,TS,bestPop,bestS,population_size,func_num,cluster_iter,dim,globalbestScore,param_bound )

                paints.plot_3d_circles_with_random_colors(TS,func_num,j, param_bound[0], param_bound[1])
                count = AloUtils.howManyGlobal(0.01,func_num,bestPop)
                cluster_count = 0

        NPF = count
        if NP == NPF:
            NSR += 1
        PR += NPF / (NP * NRs)
        print(str(PR))
    SR = NSR / NRs
