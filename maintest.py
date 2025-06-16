from DifferentialEvolution4SeachParam import *
from AloUtils import *
import selectAnother
param_num = 2
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
f10 = [0,1,0,1,12,10]
f11 = [-5,5,-5,5,6,11]
f12 = [-5,5,-5,5,8,12]
f13 = [-5,5,-5,5,6,13]
N = 0.4

#         4: himmelblau,
#         5: six_hump_camel_back,
#         6: shubert,
#         7: vincent,

#         10: modified_rastrigin_all,
#         11: CF1,
#         12: CF2,
#         13: CF3,

if __name__ == '__main__':

    NRs = 50
    MaxFEs = iter_num
    NPFs = 0
    PR = 0
    NSR = 0
    FES = 0
    param_bound = f13
    NP = param_bound[4]
    func_num = param_bound[5]
    import test


    for NR in range(50):
        TS = np.empty((4,0))
        k_count = 1
        all_population, all_scores = run(param_bound[5],population_size,top,param_num,param_bound,DevelopF,DevelopCR,iter_num,ExploreF,ExploreCR)
        bestPop,bestS = AloUtils.findallbest(all_population,all_scores)

        MulimodalPopulation, range_step, MulimodalScores = Min(all_population, param_bound[5], all_scores, k_count,param_bound)
        selectParam = 0.5
        rad = 0.05
        selectMountPop, selectMountS = dataSelect(MulimodalPopulation, range_step, MulimodalScores, param_bound,
                                                  param_bound[5], selectParam,0,NR)
        bestPopArray1, TS, clusters = detailedSearch(param_bound[5], selectMountPop, selectMountS, param_bound, rad, TS)

        # count = howManyGlobal(0.001, param_bound[5], bestPopArray1)
        selectParam = 0.7
        rad = 0.1
        selectMountPop, selectMountS = dataSelect(MulimodalPopulation, range_step, MulimodalScores, param_bound,
                                                  param_bound[5], selectParam,0.5,NR)
        bestPopArray2, TS, clusters = detailedSearch(param_bound[5], selectMountPop, selectMountS, param_bound, rad, TS)
        allbestPop = MinDE.savePopulation_Scores(bestPopArray1, bestPopArray2)
        # count = howManyGlobal(0.001, param_bound[5], allbestPop)

        selectParam = 0.9
        selectMountPop, selectMountS = dataSelect(MulimodalPopulation, range_step, MulimodalScores, param_bound,
                                                  param_bound[5], selectParam,0.7,NR)
        #使用RSII给后面的计算
        for i in range(len(selectMountPop)):
            selectMountPopparambound = [selectMountPop[i][0] - 0.1,selectMountPop[i][0] + 0.1,selectMountPop[i][1] - 0.1,selectMountPop[i][1] + 0.1,]
            data = test.RSCMSAESII(func_num, 2, selectMountPopparambound, 1000)
            dummyArchive = data['archive'].dummyArchive
            x = dummyArchive.solution
            if len(x) != 0:
                print(dummyArchive.solution)
        allbestPop1,allbestS1 = findallbest(allbestPop,cec2013main(allbestPop,func_num))
        # TS = AloUtils.bestArrayBound(allbestPop,TS)
        count = howManyGlobal(0.001, func_num, allbestPop1)
        allbestPop2 = getParam(allbestPop1,func_num,param_bound,param_num,population_size)
        count = howManyGlobal(0.001,func_num,allbestPop2)

    # selectParam = 0.3
        # selectParamStart = 0
        # rad = 0.05
        # selectMountPop, selectMountS = dataSelect(MulimodalPopulation, range_step,MulimodalScores, param_bound,param_bound[5], selectParam,selectParamStart,NR)
        #
        # bestPopArray1, TS, clusters = detailedSearch(param_bound[5], selectMountPop, selectMountS, param_bound,rad,TS)
        # howManyGlobal(0.001, param_bound[5], bestPopArray1)
        # #
        # #
        # selectParamStart = selectParam
        # selectParam = 0.5
        # rad = 0.03
        # selectMountPop, selectMountS= dataSelect(MulimodalPopulation, range_step,MulimodalScores, param_bound,param_bound[5], selectParam,selectParamStart,NR)
        # #
        # bestPopArray2,TS,clusters = detailedSearch(param_bound[5], selectMountPop, selectMountS, param_bound,rad,TS)
        # # # allbestPop = MinDE.savePopulation_Scores(bestPopArray1, bestPopArray2)
        # # # count = howManyGlobal(0.001, param_bound[5], allbestPop)
        # #
        # # selectParamStart = selectParam
        # # selectParam = 0.7
        # # rad = 0.02
        # # selectMountPop,selectMountS = dataSelect(MulimodalPopulation,range_step,MulimodalScores,param_bound,param_bound[5],selectParam,selectParamStart,NR)
        #
        # selectParamStart = 0.7
        # selectParam =0.95
        # rad = 0.01
        # selectMountPop, selectMountS = dataSelect(MulimodalPopulation, range_step, MulimodalScores, param_bound,
        #                                           param_bound[5], selectParam, selectParamStart,NR)

        # selectround(selectMountPop,selectMountS,func_num)




        # bestPopArray3,TS,clusters = detailedSearch(param_bound[5],selectMountPop,selectMountS,param_bound,rad,TS)
        #
        # allbestPop = MinDE.savePopulation_Scores(allbestPop,bestPopArray3)
        #
        # # TS = AloUtils.bestArrayBound(allbestPop,TS)
        # count = howManyGlobal(0.001, func_num, allbestPop)
        # bestPop = np.array([[ 1.75774374,1.59573728],
        #                      [-3.39510765,-3.31732141],
        #                      [-1.56154545,4.40001957],
        #                      [ 4.14126349,2.4770114 ]])
        #
        # TS = np.array([[-4,-3,-4,-3],[0,1,3,4],[1,2,1,2],[-2,-1,4,5]])
        # bestS = np.array([[0],[0],[0],[0]])
        # rebackPop = selecthiv(selectMountPop,selectMountS,bestPop,bestS,func_num)
        # selectLastPop(rebackPop,TS,func_num,param_num,population_size,param_bound,bestPop)

        # DE_iter = 4000
        # for i in range(10):
        #     MaybestPopArray,currentpop= selectAnother.selectDistant(population_size,param_bound,func_num,DE_iter,TS,allbestPop)
        #     TS = AloUtils.bestArrayBound(MaybestPopArray,TS)
        #     TS = AloUtils.bestArrayBound(currentpop,TS)
        #     allbestPop = MinDE.savePopulation_Scores(allbestPop,MaybestPopArray)
        #     count = howManyGlobal(0.001, func_num, allbestPop)
        #     if count == 6:
        #         break

    #     print('TS:' + str(TS))
        NPF = count
        if NP == NPF:
            NSR += 1
        PR += NPF / (NP * NRs)
        print(str(PR))
    SR = NSR / NRs
    #
    # print("PR=" + str(PR) + ",SR=" + str(SR) )
    # file2 = open('E:/test01/mulPOP.txt', mode='a+')
    # file2.write("F" + str(param_bound[5]) + ":PR = " + str(PR) + ",SR = " + str(SR))
    # # bestPopDistantArray = selectAnother.selectDistant(selectdistantPop,selectdistantS,param_bound,param_bound[5],1000)
    # # allbestPop = MinDE.savePopulation_Scores(bestPopArray,bestPopDistantArray)
    # # howManyGlobal(0.001,param_bound[5],allbestPop)