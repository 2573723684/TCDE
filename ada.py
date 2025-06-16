def dtypeOfPeak(selectBound,x,Max,Min):
    for i in range(len(selectBound)):
        if selectBound[i][2] >= x and selectBound[i][2] <= Max:
            selectBound[i][2] = selectBound[i][2] * 1.1
        if selectBound[i][2] <= x and selectBound[i][2] >= Min:
            selectBound[i][2] = selectBound[i][2] * 0.9
    return selectBound

def judgeByRegion(selectBound,bestPop,param_bound):
    x = (param_bound[1] - param_bound[0]) * 0.1
    Max = (param_bound[1] - param_bound[0]) * 0.2
    Min = (param_bound[1] - param_bound[0]) * 0.02
    x_bound = [param_bound[0], (param_bound[1] + param_bound[0])/2,param_bound[1]]
    y_bound = [param_bound[2], (param_bound[2] + param_bound[3])/2,param_bound[3]]
    v = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    num_bestPop = len(bestPop)
    for i in range(len(bestPop)):
        if bestPop[i][0] >= x_bound[0] and bestPop[i][1] <= x_bound[1] and bestPop[i][1] >= y_bound[0] and bestPop[i][1] <= y_bound[1]:
            x1.append(bestPop[i][0])
            x1.append(bestPop[i][1])
        if bestPop[i][0] >= x_bound[1] and bestPop[i][1] <= x_bound[2] and bestPop[i][1] >= y_bound[0] and bestPop[i][1] <= y_bound[1]:
            x2.append(bestPop[i][0])
            x2.append(bestPop[i][1])
        if bestPop[i][0] >= x_bound[0] and bestPop[i][1] <= x_bound[1] and bestPop[i][1] >= y_bound[1] and bestPop[i][1] <= y_bound[2]:
            x3.append(bestPop[i][0])
            x3.append(bestPop[i][1])
        if bestPop[i][0] >= x_bound[1] and bestPop[i][1] <= x_bound[2] and bestPop[i][1] >= y_bound[1] and bestPop[i][1] <= y_bound[2]:
            x4.append(bestPop[i][0])
            x4.append(bestPop[i][1])
    import numpy as np
    if len(x1)/2 >= num_bestPop:
        selectBound1 = np.empty((int((len(x1)/2)),2))
        count = 0
        for i in range(len(x1)):
            selectBound1[i][0] = x1[count]
            count += 1
            selectBound1[i][1] = x1[count]
            count += 1
        selectBound1 = dtypeOfPeak(selectBound1,x,Max,Min)

    elif len(x2)/2 >= num_bestPop:
        selectBound1 = np.empty((int((len(x2)/2)),2))
        count = 0
        for i in range(len(x2)):
            selectBound1[i][0] = x2[count]
            count += 1
            selectBound1[i][1] = x2[count]
            count += 1
        selectBound1 = dtypeOfPeak(selectBound1, x, Max, Min)
    elif len(x3)/2 >= num_bestPop:
        selectBound1 = np.empty((int((len(x3)/2)),2))
        count = 0
        for i in range(len(x3)):
            selectBound1[i][0] = x3[count]
            count += 1
            selectBound1[i][1] = x3[count]
            count += 1
        selectBound1 = dtypeOfPeak(selectBound1, x, Max, Min)
    elif len(x4)/2 >= num_bestPop:
        selectBound1 = np.empty((int((len(x4)/2)),2))
        count = 0
        for i in range(len(x4)):
            selectBound1[i][0] = x4[count]
            count += 1
            selectBound1[i][1] = x4[count]
            count += 1
        selectBound1 = dtypeOfPeak(selectBound1, x, Max, Min)
    else:
        selectBound1 = np.empty((0,5))
    if len(selectBound1) == 0:
        return selectBound
    else:
        for i in range(len(selectBound)):
            for j in range(len(selectBound1)):
                if selectBound[i][0] == selectBound1[j][0] and selectBound[i][1] == selectBound1[j][1]:
                    selectBound[i] = selectBound1[j]
        return selectBound


