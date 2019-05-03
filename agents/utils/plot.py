import matplotlib.pyplot as plt


def save_plot(name, xlabel, ylabel, y, x=None):
    plt.title(name)
    if x:
        plt.plot(x, y)
    else:
        plt.plot(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("./tmp/{}.png".format(name))
    plt.clf()
