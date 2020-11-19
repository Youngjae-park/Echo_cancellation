import utils
import nlms_filter as nf

import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

#Load data
x_data, x_sr = utils.load_data('speaker')
v_data, v_sr = utils.load_data('mic')

#Parameters
n_trial = 100
sig_len = x_data.shape[0]
filter_len = 1024
mu_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
delta_list = [3.0, 5.0, 0.5]


def nlms(mu, delta):
    print("mu: %.2f delta: %.2f"%(mu, delta))
    #Initialize nlms filter
    Avg_err_sqrt = np.zeros((sig_len,1)); #print(Avg_err_sqrt.shape) # (127594,1)
    
    #Adapting
    for it in tqdm(range(n_trial), mininterval=1):
        y = np.zeros((sig_len,1))
        e = np.zeros((sig_len,1))

        u = np.zeros((filter_len,1))
        w = np.zeros((filter_len,1))
        norm = 0

        for n in range(sig_len):
            norm = norm - u[-1]**2
            u[1:] = u[:-1]
            u[0] = x_data[n]
            norm = norm + u[0]**2

            y[n] = np.dot(w.T,u)
            e[n] = v_data[n] - y[n]
            w = w + (mu/(norm+delta))*u*np.conjugate(e[n])
            #"""
            #print('u => {}'.format(u))
            #print('norm => {}'.format(norm))
            #print('w => {}'.format(w))
            #"""
        Avg_err_sqrt += e**2
        #print('Avg_error => {}'.format(Avg_err_sqrt))

    Avg_err_sqrt = Avg_err_sqrt / n_trial
    
    #Write new wav file
    utils.write_data('voice'+str(mu)+'_'+str(delta)+'.wav', e, 16000)

if __name__ == '__main__':
    for m in mu_list:
        for d in delta_list:
            nlms(m,d)
