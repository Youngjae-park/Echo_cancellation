import utils
import nlms_filter as nf

import numpy as np
import matplotlib.pylab as plt

#Load data
x_data, x_sr = utils.load_data('speaker')
v_data, v_sr = utils.load_data('mic')

#Initialize nlms filter
f = nf.NLMS()
adapt = False
last = x_data.shape[0] - 1024
print("last idx is %d"%last)
hist = 0
for i in range(last+1):
    print("#%d"%i)
    px_data = x_data[i:i+1024]
    pv_data = v_data[i:i+1024]
    
    if adapt:
        f.adapt(pv_data, px_data)
    else:
        y, e, hist = f.run(pv_data, px_data)
        #print('y: {}'.format(y))
        #print('error: {}'.format(e))

print(hist)


"""
iter_num = (x_data.shape[0]//1024)
for it in range(iter_num+1):
    if it==iter_num:
        px_data = x_data[it*1024:]
        pv_data = v_data[it*1024:]
    else:
        px_data = x_data[it*1024:(it+1)*1024]
        pv_data = v_data[it*1024:(it+1)*1024]
"""
"""
print(f.mu)
print(f.n)
print(f.delta)
print(len(x_data))
print(v_data.shape)
"""


y, e, w = f.run(v_data, x_data)

print(y, e, w)
