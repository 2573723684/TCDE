# import AloUtils
# x = [1.7577346764832535, 1.596256585931725]
# x1 = [1.75780142,1.59686818]
# # x2 =  [2.1634,1.75564868]
# # x3 =  [ 4.14126349,2.4770114 ]
# # x4 = [-0.4996950653830565,-4.0125970848022625]
# # x5 = [-2.184984351133831,1.687053938819075]
# s2 = 0.000001
# # y = [2.2145448 ,1.62638307]
# #
# def panduanbestPop(bestPop,bestS,currentpop,currents,globalbestScore):
#     x = currents - globalbestScore
#     if abs(currents[0] - globalbestScore) <= 0.001:
#         counts = 0
#         for i in range(len(bestPop)):
#             if abs(currentpop[0][0] - bestPop[i][0]) <= 0.1 and abs(currentpop[0][1] - bestPop[i][1]) <= 0.1:
#                 counts += 1
#         if counts == 0:
#             bestPop = np.concatenate((bestPop, currentpop), axis=0)
#             bestS = np.concatenate((bestS, currents), axis=0)
#     return bestPop,bestS
# #
#
#
#
# import numpy as np
#
# # currentpop = np.array([-1.5610785808451173, 4.3996478856139305])
# # currents = np.array([-0.00051046])
# # bestPop = np.array([[1.757884053719756, 1.5956512405214252],[4.141315516375192, 2.4770557261123836],[-1.60028414,4.33995307]])
# # bestS = AloUtils.cec2013main(bestPop,13)
# # globalbestScore = bestS[1]
# # # a = [-1.56154545 ,4.40001957]
# # m = panduanbestPop(bestPop,bestS,currentpop,currents,globalbestScore)
# # print(m)
# from AloUtils import *
# x = [-2.19094947,1.67591216]
# DF = 0.1
# CR = 0.9
# func_num = 13
# y = [-2.184984351133831,1.687053938819075]
# TS = []
# top = 0.1
# param_num = 2
#
# selectBound = np.array([[-0.49975845,-3.99897909,0.17374624 ,-0.6721891  ,-3.9776384 ]])
# population_size = 200
# bestTopPopulation = np.empty([int(population_size * top), param_num], dtype=float)
# for i in range(len(bestTopPopulation)):
#     bestTopPopulation[i][0] = np.random.uniform(np.random.uniform(selectBound[0][0] - selectBound[0][2],selectBound[0][0] + selectBound[0][2]))
#     bestTopPopulation[i][1] = np.random.uniform(np.random.uniform(selectBound[0][1] - selectBound[0][2],selectBound[0][1] + selectBound[0][2]))
# pop,sco = selectBoundInit(selectBound[0][0],selectBound[0][1],selectBound[0][2],population_size,func_num)
# # pop[0][0] = ParametersPeak[cluster][0]
# # pop[0][1] = ParametersPeak[cluster][1]
# # sco[0] = ParametersPeak[cluster][7]
# cluster_iter = 5000
# for it in range(cluster_iter):
#     mutated_population = searchDmutate(pop,DF,bestTopPopulation)
#     crossed_population = searchcross(mutated_population,pop,CR)
#     selectpop,selectsco = searchselect(crossed_population,pop,sco,func_num,TS,selectBound[0][0],selectBound[0][1],selectBound[0][2])
#     pop = np.copy(selectpop)
#     sco = np.copy(selectsco)
#     index = desort(selectsco)
#     for i in range(len(bestTopPopulation)):
#         bestTopPopulation[i][0] = selectpop[int(index[i])][0]
#         bestTopPopulation[i][1] = selectpop[int(index[i])][1]
#
#     current_poplist = [selectpop[int(index[0])][0],selectpop[int(index[0])][1]]
#     current_slist = selectsco[int(index[0])]
#     print(current_slist)
# # print(str(selectBound[0]) +',迭代次数:'+str(it)+'最优解:'+str(current_poplist)+':'+ str(current_slist))
import numpy as np
x = np.array([[0.62422843,0.62422843],[0.33301844 ,0.33301844],
 [7.70627726 ,7.70627726],
 [4.11120715 ,7.70627726],
 [7.70627726, 2.19328005],
 [4.11120714 ,4.11120715],
 [7.70627726 ,1.17008879],
 [2.19328005 ,4.11120715],
 [0.62422843 ,4.11120714],
 [4.11120714 ,2.19328005],
 [2.19328005 ,1.17008879],
 [1.17008879, 7.70627726],
 [7.70627724 ,0.62422843],
 [2.19328005, 2.19328005],
 [4.11120714 ,0.62422843],
 [1.17008879 ,4.11120714],
 [0.62422843, 7.70627726],
 [0.33301844 ,7.70627726],
 [0.62422843, 2.19328005],
 [1.17008879, 0.62422843],
 [0.62422843 ,0.33301844],
 [0.33301844 ,1.17008879],
 [1.17008879, 0.33301844],
 [0.33301844, 2.19328005],
 [2.19328005 ,0.62422843],
 [0.62422843, 1.17008879],
 [2.19328005 ,0.33301844],
 [0.33301844 ,4.11120714],
 [7.70627725, 0.33301844],
 [4.11120714 ,0.33301844],
 [1.17008879 ,2.19328005],
 [1.17008879, 1.17008879],
 [7.70627727 ,4.11120715],
 [4.11166574, 1.17012561],
 [2.19402567,7.69650408]])
import matplotlib.pyplot as plt
pop1 = []
pop2 = []
for i in range(len(x)):
    pop1.append(x[i][0])
    pop2.append(x[i][1])
plt.scatter(pop1, pop2, c='b', s=20)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.title("TSall_point")
plt.xlabel("x")
plt.ylabel("y")
plt.show()