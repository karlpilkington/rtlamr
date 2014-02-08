import matplotlib.pyplot as plt

import numpy as np

width = 800
phi = 1.61803398875
dpi = 96.0

sample_rate = 2.4e6
data_rate = 32.768e3
symbol = sample_rate / data_rate

raw = np.memmap("mag.bin", dtype=np.float64, mode='r')

raw = np.roll(raw, 4096)
raw = raw[3500:19500]
raw /= raw.max()

fig, subplots = plt.subplots(nrows=2)

fig.set_size_inches(width/dpi,(width/phi)/dpi)
plt.tight_layout()

for plot in subplots.flat:
	plot.autoscale(enable=True, tight=True)
	plot.grid()

(mag_plot, zoom) = list(subplots.flat)

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

one = np.ones((symbol*2,))
one[symbol:] = -1

corr = np.correlate(raw, one)

corr /= np.abs(corr).max()

mag_plot.plot(corr, linewidth=0.5)
mag_plot.set_xlim(0, corr.size)

zoom.plot(np.arange(raw.size) - symbol, raw, linewidth=0.5)
zoom.plot(corr, linewidth=1.25)
# zoom.fill(corr, color='green', alpha=0.5)

low = align - symbol*4
high = align + symbol*24
mag_plot.axvspan(low, high, color='grey', alpha=0.25)
zoom.set_xlim(low, high)

for idx in np.arange(0,192*symbol, symbol*2) + align:
	zoom.axvline(idx, color='red')

zoom.set_ylim(-1.125, 1.125)

fig.savefig('../matched_filter.svg', dpi=dpi, transparent=True)
plt.show()