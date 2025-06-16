""" A simple example to show how to perform multimodal optimization using this code

Written by Ali Ahrari
last update on 17 Jan 2022
For all inquiries, please contact Ali Ahrari (aliahrari1983@gmail.com) """


def driver(opt, problem):
    """ Solve the multimodal optimization problem given the problem and optimization options """
    startTime = time.time()
    process = OptimProcess(opt, problem)  # create the object process
    archive = Archive(problem)  # create the object archive
    while (process.usedEvalTillRestart < problem.maxEval):
        restart = Restart(process, opt, problem)  # create the object restart (restart-specific information)
        subpop = restart.initialize_subpop(archive, process, opt, problem)  # initialize the subpopulation
        restart.run_one_restart(subpop, archive, process, opt, problem)  # evolve the subpopulation until convergence
        if (restart.terminationFlag == -1):  # evaluation budget finished
            break
        archive.update(subpop, restart, process, opt, problem)  # update archive
        process.update(restart, archive, opt, problem)  # update process
        print('restartNo = ' + str(process.restartNo - 1), ', usedEval = ' +
              str(process.usedEvalTillRestart) + '(' + '{:.2f}'.format(
            process.usedEvalTillRestart / problem.maxEval * 100) + '%)',
              ', archiveSize = ' + str(archive.size))  # print optimization progress summary after each restart
    data = {'archive': archive, 'process': process, 'usedTimeSec': time.time() - startTime}  # store archive and process
    return data


# --------------- Import packages ------------------
import sys
import os

cwd = os.getcwd()  ## gets the path where the file is
sys.path.append(cwd + '/cec2013')
sys.path.append(cwd + '\cec2013')
import numpy as np
import time
import pandas as pd
from RSCMSAESII_v1.OptimProblem import OptimProblem
from RSCMSAESII_v1.OptimOption import OptimOption
from RSCMSAESII_v1.OptimProcess import OptimProcess
from RSCMSAESII_v1.Archive import Archive
from RSCMSAESII_v1.Restart import Restart



def RSCMSAESII(func_num,param,param_bound,iter_num):
    # ------------------ problem and run settings ---------------------
    problemID = func_num  # ID for the optimization problem (ID=1, 2, ..., 20) corresponds to the well-known CEC'2013
    # test suite for multimodal optimization; however, you should download the relevant files first.
    # ID=21 is the Vincent function manually defined
    dim = param # problem dimensionality
    seedNo = 0  # 随机生成种子

    # ---------------- 初始化问题和选项对象 -------------------
    startTime = time.time()  # 开始时间
    problem = OptimProblem(problemID, dim)  # 创建问题对象
    problem.set_problem_data(iter_num,param_bound)  # set problem information
    opt = OptimOption(problem)  # create the optimization object
    np.random.seed(seedNo)  # random seed number

    data = driver(opt, problem)  # run the optimization process, and store detailed information in the dictionary data
    return data
    # dummyArchive stores data of the reported solutions, and the time/evaluation that they have been found,
    # and the code (-1,0,1). Code -1 means that the solution should be removed from the report, and code 1 means
    # the solution should be added to the report.
    # dummyArchive = data['archive'].dummyArchive  # extract dummmyArchive from data

    # # The rest of this code writes the required information for performance evaluation in a CSV file. All this information is stored in dummyArchive
    # output = np.zeros((dummyArchive.solution.shape[0], problem.dim + 4))
    # output[:, 0:problem.dim] = dummyArchive.solution  # reported solutions
    # output[:, problem.dim] = dummyArchive.value  # reported solutions
    # output[:, problem.dim + 1] = dummyArchive.foundEval  # evalaution number at which the solutions were found
    # output[:, problem.dim + 2] = dummyArchive.foundTime  # the CPU used time to find each solution (ms)
    # output[:, problem.dim + 3] = dummyArchive.actionCode  # Action code for the reported solution (1: add, -1: remove)
    #
    # colNames = ['x1']  # column names
    # for k in np.arange(problem.dim - 1):
    #     colNames.append('x' + str(k + 2))
    # colNames = colNames + ['f', 'foundEval', 'foundTime (ms)', 'actionCode']
    #
    # # Write output to a csv file
    # outputDF = pd.DataFrame(output, columns=colNames)
    # outputDF.to_csv(('result_' + str(seedNo) + '.csv'), index=False)
    #
    # print('\nRun completed successfully after ' + '{:.2f}'.format(
    #     time.time() - startTime) + ' seconds. \nSee the result file (result_xx.csv)')
