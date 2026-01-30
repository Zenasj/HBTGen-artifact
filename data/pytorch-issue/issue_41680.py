#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

figure = plt.figure()
plt.axis('scaled')
plt.tight_layout()
matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
plt.close(figure)
print("Matplotlib version:", matplotlib.__version__)
assert plt.fignum_exists(figure.number) == False