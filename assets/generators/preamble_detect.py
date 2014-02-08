import matplotlib.pyplot as plt

import numpy as np

width = 800
phi = 1.61803398875
dpi = 84.0

sample_rate = 2.4e6
data_rate = 32.768e3
symbol = sample_rate / data_rate

raw = np.memmap("mag.bin", dtype=np.float64, mode='r')

raw = np.roll(raw, 4096)
raw = raw[:25000]
raw /= raw.max()

fig, subplots = plt.subplots(nrows=3)

fig.set_size_inches(width/dpi,(width/phi)/dpi)
plt.tight_layout()

for plot in subplots.flat:
	plot.autoscale(enable=True, tight=True)
	plot.grid()

(mag_plot, corr_plot, zoom) = list(subplots.flat)

one = np.ones((146,))
one[73:] = -1

preamble = 0x1F2A60
bits = bin(preamble)[2:]
preamble_len = len(bits)*symbol*2

preamble_sig = np.zeros(preamble_len)
idxs = np.arange(0, preamble_len, symbol*2)
for bit, idx in zip(bits, idxs):
	if bit == '1':
		preamble_sig[idx:idx+symbol] = 1
		preamble_sig[idx+symbol:idx+symbol*2] = -1
	else:
		preamble_sig[idx:idx+symbol] = -1
		preamble_sig[idx+symbol:idx+symbol*2] = 1

preamble_sig[0] = -1
preamble_sig[-1] = -1

corr = np.correlate(raw, preamble_sig)
align = corr.argmax()

mag_plot.plot(raw, linewidth=0.5)
zoom.step(np.arange(preamble_sig.size)+align, (preamble_sig+1)/2, color='red')
mag_plot.set_ylim(-0.125, 1.125)

corr /= np.abs(corr).max()

corr_plot.plot(corr)
corr_plot.set_ylim(-0.5,1.0)

low = align
high = align + preamble_sig.size

mag_plot.axvline(low, color='red')
mag_plot.axvline(high, color='red')
mag_plot.axvspan(low, high, color='grey', alpha=0.25)

corr_plot.axvline(low, color='red')
corr_plot.axvline(high, color='red')
corr_plot.axvspan(low, high, color='grey', alpha=0.25)

zoom.plot(raw, linewidth=0.5)
zoom.plot(corr, linewidth=1.25)
zoom.set_xlim(low-symbol*2, high+symbol*2)
zoom.set_ylim(-0.75,1.25)

mag_plot.set_xlim(0, corr.size)
corr_plot.set_xlim(0, corr.size)

fig.savefig('../preamble_detect.svg', dpi=dpi, transparent=True)
plt.show()