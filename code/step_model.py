import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
matplotlib.rc('text', usetex=True)
import numpy as np
from fakedata import *
from scipy.optimize import fmin

def step(width, t0, height, rate, data):
    model = len(data) * [rate]
    model = np.array(model)
    left = t0 < data
    right = data < t0+width
    box = left * right
    model[box] = height
    return model

def ln_like(pars, data):
    width, tstep = pars
    data = np.array(sorted(data))
    w = width
    t0 = tstep-data[0]
    t1 = data[-1] - (tstep + w)
    Nin = sum((tstep < data) * (data < tstep+w))
    Nout = len(data) - Nin
    h = Nin/w
    b = Nout/(t0+t1)
    return Nout*np.log(b) + Nin*np.log(h) - b*(t0+t1) - h*w

def get_hb(pars,data):
    width, tstep = pars
    data = np.array(sorted(data))
    w = width
    t0 = tstep-data[0]
    t1 = data[-1] - (tstep + w)
    Nin = sum((tstep < data) * (data < (tstep+w)))
    Nout = len(data) - Nin
    h = Nin/w
    b = Nout/(t0+t1)
    return h,b

if __name__ == "__main__":    
    width = 6
    tstep = 47

    plt.figure(figsize = (8,12))
    fakedata = photons(0.1, 100, 1, 1, 100, 50)
    print len(fakedata)
    fakedata = np.array(fakedata)
    plt.subplot(411)
    lls = []
    widths = np.arange(1, 25, .1)
    #optimization test
    pars = [width, tstep] 
    opt = fmin(ln_like, x0=pars, args=([fakedata]))
    print opt[0], opt[1]
    pars = [opt[0], opt[1]]
    #print fmin(get_hb, x0=pars, args=([fakedata]))
    print get_hb(pars, fakedata)
    assert False

    for w in widths:
        ll = ln_like(w, tstep, fakedata)
        lls.append(ll)

    plt.xlabel('step width')
    plt.ylabel('likelihood')
    plt.plot(widths, lls)
    ymin = np.sort(lls)[np.isfinite(np.sort(lls))][0]
    ymax = np.sort(lls)[np.isfinite(np.sort(lls))][-1]
    plt.vlines(x=width, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.subplot(412)
    lls = []
    bs = []
    heights = []
    tsteps = np.arange(0,100,.1)

    for t in tsteps:
        ll = ln_like(width, t, fakedata)
        lls.append(ll)

        b = get_hb(width, t, fakedata)[1]
        bs.append(b)

        height = get_hb(width, t, fakedata)[0]
        heights.append(height)

    plt.plot(tsteps,lls)
    ymin = np.sort(lls)[np.isfinite(np.sort(lls))][0]
    ymax = np.sort(lls)[np.isfinite(np.sort(lls))][-1]
    plt.vlines(x=tstep, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.xlabel('step start time')
    plt.ylabel('likelihood')

    plt.subplot(413)
    hminusb = [h-b for h,b in zip(heights,bs)]
    plt.plot(tsteps,hminusb)
    ymin = np.sort(hminusb)[np.isfinite(np.sort(hminusb))][0]
    ymax = np.sort(hminusb)[np.isfinite(np.sort(hminusb))][-1]
    plt.vlines(x=tstep, ymin=ymin, ymax=ymax, colors='red', linestyle='--')
    plt.xlabel(r'$t_{step}$')
    plt.ylabel('h-b (step height)')

    plt.subplot(414)
    plt.hist(fakedata, bins=len(fakedata), alpha = 0.5)
    plt.tight_layout()

    plt.savefig('step_lnll.pdf')


