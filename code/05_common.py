import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)

def plot(x, ys, lbls=None, markers=None, xlabel='Time', ylabel='a.u.', path=None, ylim=None):
    plt.figure(figsize=(6,4))

    for i, y in enumerate(ys):
        if lbls: lbl = lbls[i]
        else: lbl = None
        if markers: plt.plot(x,y,markers[i],label=lbl)
        else: plt.plot(x,y,label=lbl)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.minorticks_on()
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle=':')
    if ylim: plt.ylim(ylim)
    plt.tight_layout()
    if lbls: plt.legend()
    
    if path: plt.savefig(path, dpi=300)
    else: plt.show()
