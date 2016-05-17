import numpy as np

m = np.genfromtxt('./mossong/italy_phy.csv', delimiter='\t')
print (m[0,0])