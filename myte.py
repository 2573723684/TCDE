def cir():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    cir = Circle((1, 1), radius=0.5)
    ax.add_patch(cir)
    plt.show()

if __name__ == '__main__':
    cir()
