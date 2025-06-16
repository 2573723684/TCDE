import AloUtils
if __name__ == '__main__':
    pop = np.empty((100,2))
    

    for i in range(len(pop)):
      pop[i][0] = np.random.uniform(-5,5)
      pop[i][1] = np.random.uniform(-5,5)
    distancendarry = AloUtils.nddistance(pop)
