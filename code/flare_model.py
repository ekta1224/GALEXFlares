import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
import matplotlib.pylab as plt
import numpy as np
from scipy.special import erf

def flare_model(start, stop, rate, sigma, amp, mu):
    gaussian = list(np.random.normal(mu ,sigma, amp))
    #print len(gaussian)
    flat = rate*list(np.arange(start,stop,0.1))
    #print len(flat)
    photons =  flat + gaussian
    return photons

def ln_likelihood(start, stop, rate, sigma, amp, mu, data):
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
    sigma = 0.5
    amp = 100.
    rate = 1
    mu = 50

    plt.figure(figsize=(8,12))    
    data = flare_model(start, stop, rate, sigma, amp, mu)

    
    plt.subplot(411)
    sigmas = np.arange(-19.5, 20.5, 0.1) 
    lls_sig = [] 
    for sig in sigmas:
        ll = ln_likelihood(start, stop, rate, sig, amp, mu, data)
        lls_sig.append(ll)
    plt.plot(sigmas, lls_sig)
    ymin = np.isfinite(np.sort(lls_sig))
    ymax = lls_sig[np.isfinite(sorted(lls_sig))][-1]
    print ymin,ymax
    assert False
    plt.vlines(x=sigma, ymin=-5000, ymax=1000, colors='red', linestyle='--')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('likelihood')

    plt.subplot(412)
    lls_mu = []
    mus = np.arange(0.0001,100, .5)
    for m in mus:
        ll = ln_likelihood(start, stop, rate, sigma, amp, m, data)
        lls_mu.append(ll)
    plt.plot(mus, lls_mu)
    plt.vlines(x=mu, ymin=-150, ymax=300, colors='red', linestyle='--')
    plt.xlabel(r'$\mu$')
    plt.ylabel('likelihood')

    plt.subplot(413)
    lls_amp = []
    amps = np.arange(1, 200, 5)
    for am in amps:
        ll = ln_likelihood(start, stop, rate, sigma, am, mu, data)
        lls_amp.append(ll)
    plt.plot(amps, lls_amp)
    plt.vlines(x=amp, ymin=-50, ymax=300, colors='red', linestyle='--')

    plt.xlabel(r'amp')
    plt.ylabel('likelihood')

    plt.subplot(414)
    plt.hist(data, bins= len(data), alpha = 0.5)
    plt.tight_layout()
    plt.savefig('gauss_lnll.pdf')


    assert False
    #make fata data by varying data = flare_model(start, stop, rate, sig, amp, mu)
    data = flare_model(start, stop, rate, sigma, amp, mu)
    plt.figure()
    plt.hist(data, bins=len(data), alpha = 0.5)
    plt.xlabel('time')
    plt.title(r'$\sigma=%s, amplitude=%s, \mu=%s$' %(sigma,amp,mu))
    plt.savefig('fake_data4.png')
    
