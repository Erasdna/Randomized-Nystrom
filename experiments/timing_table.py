import numpy as np
import os 

sketches = ["Gaussian", "SASO", "LASO", "SSO"]
sizes= ["3000","9000","15000"]

for sketch in sketches:
    for size in sizes:
        filename = os.getcwd() + "/Figures/Timing/" + sketch + "/Strong_n=" + size + "_data.txt"
        dat = np.loadtxt(filename)[:,1:]
        base = dat[0,:]
        mean = base[0] / dat[1:,0]
        std = np.abs((dat[1:,1]*base[1] - base[0]*dat[1:,0])/base[0]**2)
        print(sketch, " ", size, ": ")
        print(base)
        print(mean)
        print(std) 