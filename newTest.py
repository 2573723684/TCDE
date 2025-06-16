from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import AloUtils
import numpy as np
def Min(all_population,all_scores,param_bound):
    length = (param_bound[1] - param_bound[0]) / 2
    for l in range(2):
        x = []
        xs = []
        x_lower = param_bound[0] + l * length
        x_upper = param_bound[0] + (l + 1) * length
        for pop in range(len(all_population)):
            if x_lower <= all_population[pop][0] < x_upper:
                x.append(all_population[pop])
                xs.append(all_scores[pop])
        for w in range(2):
            xy = []
            xys = []
            y_lower = param_bound[0] + w * length
            y_upper = param_bound[0] + (w + 1) * length
            for pop in range(len(x)):
                if y_lower <= x[pop][1] < y_upper:
                    xy.append(x[pop])
                    xys.append(xs[pop])
            for h in range(2):
                xyz = []
                xyzs = []
                z_lower = param_bound[0] + h * length
                z_upper = param_bound[0] + (h + 1) * length
                for pop in range(len(xy)):
                    if z_lower <= pop[2] < z_upper:
                        xyz.append(xy[pop])
                        xyzs.append(xys[pop])
                params = [x_lower,x_upper,y_lower,y_upper,z_lower,z_upper]



if __name__ == '__main__':

    import AloUtils
    Pop = np.array([[1,2,3]])
    c = AloUtils.howManyGlobal(0.001,8,Pop)