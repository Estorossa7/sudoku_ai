import matplotlib.pyplot as plt
import numpy as np

def plot(x,y,name):
    plt.plot(x,label=name)
    plt.plot(y, label=f"mean_{name}" )
    plt.title(f"{name} over episods")
    plt.show()