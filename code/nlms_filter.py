import numpy as np

class Af(): #Adaptivefilter
    def init_weights(self, mode, n=-1):
        if n == -1:
            n = self.n
        if mode == 'random':
            w = np.random.normal(0, 0.5, n)
        elif mode == 'zeros':
            w = np.zeros(n)
        w = np.array(w, dtype="float64")
        self.w = w

class NLMS(Af):
    def __init__(self, n=1024, mu=0.1, delta=1., weights='zeros'):
        self.n = n # Filter length
        self.mu = mu
        self.delta = delta
        self.init_weights(weights, self.n) #weights in filters can be initialized various way. in my code, it only have random and zero initializing methods.
        self.w_history = False

    def adapt(self, d, x):
        y = np.dot(self.w, x) #output signal
        e = d - y #error = desired signal - output signal
        #If np.dot(x,x) is very small, then the problem can be occured because denominator is zero. so delta value is need to avoid upper problem. 
        nu = self.mu / (self.delta + np.dot(x, x))
        self.w += nu*e*x

    def run(self, d, x):
        N = len(x)
        if not len(d) == N:
            print('the length of desired signal and matrix x is not same')
        self.n = x.shape[0] #len(x[0])
        x = np.array(x)
        d = np.array(d)

        y = np.zeros((N,N))
        e = np.zeros((N,N))
        self.w_history = np.zeros((N, self.n))
        
        # adaptation loop
        for k in range(N):
            self.w_history[k,:] = self.w
            #print(self.w, self.w.shape)
            #print(x[k], x[k].shape)
            #print(y[k])
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            nu = self.mu / (self.delta + np.dot(x[k], x[k]))
            dw = nu*e[k]*x[k]
            self.w += dw

        return y, e, self.w_history

"""
if __name__ == '__main__':
    a = NLMS(n=1024, mu=0.1, weights='random')
#"""
