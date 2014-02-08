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
raw = raw[:25000]
raw /= raw.max()

fig, subplot = plt.subplots(nrows=1)

fig.set_size_inches(width/dpi,(width/phi**2)/dpi)
plt.tight_layout()

symlen = int(symbol)
subplot.stem(np.append(np.ones(symlen), -np.ones(symlen)), markerfmt='.')
subplot.set_xlim(-2,symlen*2+1)
subplot.set_ylim(-1.125,1.125)
subplot.grid()

fig.savefig('../basis.svg', dpi=dpi, transparent=True)

plt.show()