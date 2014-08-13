import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
import matplotlib.pylab as plt
import numpy as np
from scipy.special import erf
from fakedata import *

def gauss_lnlike(start, stop, rate, sigma, amp, mu, data):
    def gamma(rate, sigma, amp, mu, data):
        gammas = []
        for d in data:
            gammas.append(rate + amp/ (sigma * np.sqrt(2*np.pi)) * np.exp(-(d-mu)**2/(2*sigma**2)))
        return gammas
   
    lnlike = np.sum(np.log(gamma(rate, sigma, amp, mu, data))) - rate*(stop-start) - (amp * np.sqrt(np.pi/2) * sigma * (erf((stop - mu)/np.sqrt(2)*sigma) - erf((start- mu)/ np.sqrt(2)*sigma)))
    return lnlike

if __name__ == "__main__":
    start = 0.1
    stop = 100
    sigma = 2.5
    amp = 100.
    rate = 1
    mu = 50

    plt.figure(figsize=(8,12))    
    data = photons(start, stop, rate, sigma, amp, mu)
    
    plt.subplot(411)
    sigmas = np.arange(-19.5, 20.5, 0.1) 
    lls_sig = [] 
    for sig in sigmas:
        ll = gauss_lnlike(start, stop, rate, sig, amp, mu, data)
        lls_sig.append(ll)
    plt.plot(sigmas, lls_sig)
    ymin = np.sort(lls_sig)[np.isfinite(np.sort(lls_sig))][0]
    ymax = np.sort(lls_sig)[np.isfinite(np.sort(lls_sig))][-1]
    print ymin, ymax
    plt.vlines(x=sigma, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('likelihood')

    plt.subplot(412)
    lls_mu = []
    mus = np.arange(0.0001,100, .5)
    for m in mus:
        ll = gauss_lnlike(start, stop, rate, sigma, amp, m, data)
        lls_mu.append(ll)
    ymin = np.sort(lls_mu)[np.isfinite(np.sort(lls_mu))][0]
    ymax = np.sort(lls_mu)[np.isfinite(np.sort(lls_mu))][-1]
    print ymin, ymax
    plt.plot(mus, lls_mu)
    plt.vlines(x=mu, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.xlabel(r'$\mu$')
    plt.ylabel('likelihood')

    plt.subplot(413)
    lls_amp = []
    amps = np.arange(1, 200, 5)
    for am in amps:
        ll = gauss_lnlike(start, stop, rate, sigma, am, mu, data)
        lls_amp.append(ll)
    ymin = np.sort(lls_amp)[np.isfinite(np.sort(lls_amp))][0]
    ymax = np.sort(lls_amp)[np.isfinite(np.sort(lls_amp))][-1]
    print ymin,ymax
    plt.plot(amps, lls_amp)
    plt.vlines(x=amp, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.xlabel(r'amp')
    plt.ylabel('likelihood')

    plt.subplot(414)
    plt.hist(data, bins= len(data), alpha = 0.5)
    plt.tight_layout()
    plt.savefig('gauss_lnll_wide.pdf')


