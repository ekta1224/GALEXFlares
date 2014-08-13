import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
matplotlib.rc('text', usetex=True)
import numpy as np
from flare_model import flare_model

def step(width, t0, height, rate, data):
    model = len(data) * [rate]
    model = np.array(model)
    left = t0 < data
    right = data < t0+width
    box = left * right
    model[box] = height
    return model

def ln_like(width, tstep, data):
    data = np.array(sorted(data))
    w = width
    t0 = tstep-data[0]
    t1 = data[-1] - (tstep + w)
    Nin = sum((tstep < data) * (data < tstep+w))
    Nout = len(data) - Nin
    h = Nin/w
    b = Nout/(t0+t1)
    return Nout*np.log(b) + Nin*np.log(h) - b*(t0+t1) - h*w

def get_hb(width, tstep, data):
    data = np.array(sorted(data))
    w = width
    t0 = tstep-data[0]
    t1 = data[-1] - (tstep + w)
    Nin = sum((tstep < data) * (data < (tstep+w)))
    Nout = len(data) - Nin
    h = Nin/w
    b = Nout/(t0+t1)
    return h,b
    

width = 6
tstep = 47
# height = 100
# rate = 1
# data = np.arange(0,100,1.)
# stepfn = step(width, tstep, height, rate, data)

# plt.figure()
# plt.plot(data,stepfn)
# plt.savefig('step_model.pdf')

plt.figure(figsize = (8,12))
fakedata = flare_model(0.1, 100, 1, 1, 100, 50)
print len(fakedata)
fakedata = np.array(fakedata)
plt.subplot(411)
lls = []
widths = np.arange(1, 25, .1)

for w in widths:
    ll = ln_like(w, tstep, fakedata)
    lls.append(ll)
#print heights, lls
#assert False
plt.xlabel('step width')
plt.ylabel('likelihood')
plt.plot(widths, lls)

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
plt.xlabel('step start time')
plt.ylabel('likelihood')

plt.subplot(413)
plt.plot(tsteps, [h-b for h,b in zip(heights,bs)])
plt.xlabel(r'$t_{step}$')
plt.ylabel('h-b (step height)')

plt.subplot(414)
plt.hist(fakedata, bins=len(fakedata), alpha = 0.5)
plt.tight_layout()

plt.savefig('step_lnll.pdf')
    
 
