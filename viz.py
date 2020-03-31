import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lines = open('viz.txt').read().strip().split('\n')
json_lines = []
for line in lines:
    if 'loss' in line:
        json_lines.append(json.loads(line.replace('\'', '"')))
d = pd.DataFrame(json_lines)
loss_columns = [a for a in list(d.columns) if 'loss' in a]
acc_columns = [a for a in list(d.columns) if 'accuracy' in a]

loss_d = d[loss_columns]
acc_d = d[acc_columns]
print(d.head())

fig, axes = plt.subplots(nrows=2, ncols=1)
loss_d.plot(ax=axes[0])
acc_d.plot(ax=axes[1], cmap='jet')
plt.show()
