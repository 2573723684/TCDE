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
f4 = [-6, 6, -6, 6, 4, 4]
f5 = [-1.9, 1.9, -1.9, 1.9, 2, 5]
f6 = [-10, 10, -10, 10, 18, 6]
f7 = [0.25, 10, 0.25, 10, 36, 7]
f8 = [-10, 10, -10, 10, 81, 8]
f9 = [0.25, 10, 0.25, 10, 216, 9]
f10 = [0, 1, 0, 1, 12, 10]
f11 = [-5, 5, -5, 5, 6, 11]
f12 = [-5, 5, -5, 5, 8, 12]
f13 = [-5, 5, -5, 5, 6, 13]
f14 = [-5, 5, -5, 5, 6, 14]
f15 = [-5, 5, -5, 5, 8, 15]
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
    param_bound = f8
    NP = param_bound[4]
    func_num = param_bound[5]

    max_population = 2000
    for NR in range(50):
        selectParam = 0.5
        radious = 0.4
        min_points = 2
        countiter = 0
        cluster_count = 0
        globalbestScore = -999999
        TS = np.empty((0, dim + 1))
        all_population = np.empty((0, dim))
        AllclusterPop = np.empty((0, dim))
        AllclusterS = np.empty((0, 1))
        bestPop = np.empty((0, dim))
        bestS = np.empty((0, 1))
        for j in range(20):
            # if j > 1:
            #     if count == param_bound[4]:
            #         break
            # if j % 1 == 0 and j > 1:
            #     selectParam -= 0.05
            #     if selectParam <= 0.1:
            #         selectParam = 0.1
            # if j % 1 == 0 and j > 1:
            #     radious -= 0.05
            #     if radious <= 0.1:
            #         radious = 0.1

            error = 0
            all_population, all_scores, countiter, current_best, current_best_scores = run(func_num, population_size,top, dim, param_bound,DevelopF, DevelopCR,iter_num, ExploreF, ExploreCR, TS, max_population,cluster_count)
            import paints
            # paints.paint(all_population, param_bound[0], param_bound[1], func_num, j)
            MulimodalPopulation, MulimodalScores = Min(all_population, func_num, all_scores, param_bound)

            if error == 0:
                selectMountPop, selectMountS = dataSelect(MulimodalPopulation, MulimodalScores, param_bound,param_bound[5], selectParam, 0, NR)
                selectMountPop, selectMountS = AloUtils.savePopulation_Scores(AllclusterPop, selectMountPop,AllclusterS, selectMountS)
                clusters, clusterPop, clusterS = dbscn(selectMountPop, radious, min_points, selectMountS)
                AllclusterPop = np.copy(clusterPop)
                AllclusterS = np.copy(clusterS)

                selectBound = baseClusterSearch(clusters, selectMountPop, selectMountS, bestPop)
                # ada.judgeByRegion(selectBound,bestPop,param_bound)

                cluster_iter = 200
                # 围绕着每个小生境内，以最优潜在峰顶为圆心，最差潜在峰顶到最优潜在峰顶的距离为半径，在这个圆内进行搜索      TS, bestPop, bestS, globalbestScore = Search(selectBound, TS, bestPop, bestS, 15, func_num,cluster_iter, dim, globalbestScore, param_bound)
                paints.plot_3d_circles_with_random_colors(TS, func_num, j, param_bound[0], param_bound[1],NR)
                count = AloUtils.howManyGlobal(0.01, func_num, bestPop)
                cluster_count = 0

        NPF = count
        if NP == NPF:
            NSR += 1
        PR += NPF / (NP * NRs)
        print(str(PR))
    SR = NSR / NRs
