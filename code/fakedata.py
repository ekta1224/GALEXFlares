import numpy as np
from scipy.special import erf

def photons(start, stop, rate, sigma, amp, mu):
    gaussian = list(np.random.normal(mu ,sigma, amp))
    #print len(gaussian)
    flat = rate*list(np.arange(start,stop,0.1))
    #print len(flat)
    photons =  flat + gaussian
    return photons

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rc('text', usetex=True)
    import matplotlib.pylab as plt

    start = 0.1
    stop = 100
    sigma = 0.5
    amp = 100.
    rate = 1
    mu = 50

    data = photons(start, stop, rate, sigma, amp, mu)
    plt.figure()
    plt.hist(data, bins=len(data), alpha = 0.5)
    plt.title(r'$\sigma = %s, amplitude = %s, \mu = %s$' %(sigma, amp, mu)) 
    plt.savefig('fakedata.pdf')
