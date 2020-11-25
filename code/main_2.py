import utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
cnt = 0


def e_graph(val, mu, delta, name='avg_err'):
    global cnt
    if name == 'avg_err':
        fig = plt.figure(cnt)
        plt.semilogy(val)
        plt.xlabel('Number of iterations')
        plt.ylabel('Ensemble-averaged square error')
        cnt += 1
        ''' 
        fig = plt.figure(cnt)
        plt.subplot(2,1,1)
        plt.plot(d)
        plt.ylim([-0.5, 0.5])
        plt.subplot(2,1,2)
        plt.plot(e)
        plt.ylim([-0.5, 0.5])
        '''

        fname = 'voice'+str(mu)+'_'+str(delta)
        fig.savefig(fname+'.png')
        return fig
    
    elif name == 'd_e':
        fig = plt.figure(cnt)
        plt.subplot(2,1,1)
        plt.plot(val[0])
        plt.ylim([-0.5,0.5])
        plt.subplot(2,1,2)
        plt.plot(val[1])
        plt.ylim([-0.5,0.5])

        fname = 'voice'+str(mu)+'_'+str(delta)
        fig.savefig(fname+'_d_e.png')
        cnt += 1
        return fig

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

    d_e = [v_data,e]
    Avg_err_sqrt = Avg_err_sqrt / n_trial
    avg_err_graph = e_graph(Avg_err_sqrt,mu,delta,'avg_err')
    d_e_graph = e_graph(d_e, mu, delta,'d_e')
    #Write new wav file
    utils.write_data('voice'+str(mu)+'_'+str(delta)+'.wav', e, 16000)

if __name__ == '__main__':
    for m in mu_list:
        for d in delta_list:
            nlms(m,d)
