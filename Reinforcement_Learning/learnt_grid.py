import numpy as np
import matplotlib.pyplot as plt


def get_colour(value):
    if value == 1:
        return 'forestgreen'
    elif value == 0.5:
        return 'lightgreen'
    elif value == -1:
        return 'firebrick'
    else:
        return 'lightcoral'


colours = ['forestgreen', 'lightgreen', 'lightcoral', 'firebrick']
labels = ['Hit', 'Maybe Hit', 'Maybe Stand', 'Stand']

fig = plt.figure(figsize = (6.5,7.5))
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, right=False, left=False)

for i in range(4):
    ax1.scatter(21, 11, c = colours[i], marker = 's', label = labels[i], s = 250)

data = np.load('.\learnt_strategy.npy'.format('30'))
# print(data)
data_ace = data[18:,:]
data_no_ace = data[:18,:] # need to plot both ace and no ace in separate tables in same figure

for x, x_vals in enumerate(data_ace):
    for y, y_vals in enumerate(x_vals):
        if x > 7:
            ax1.scatter(x+4, y+2, c = get_colour(y_vals), marker = 's', s = 250)
        else:
            ax1.scatter(x+4, y+2, c = 'white', marker = 's', s = 250)

ax.set_xlabel('Player Total', fontsize=12)
ax.set_ylabel('Dealer Card', fontsize=12)
ax1.set_title('Usable Ace', loc='left', fontsize=15, weight='bold')
ax1.set_xticks(np.arange(4,22,1))
ax1.set_yticks(np.arange(2,12,1))
ax1.legend(loc = 'upper center', bbox_to_anchor = (0.7, 1.3), ncol = 2, fancybox = True, shadow = True)

for x, x_vals in enumerate(data_no_ace):
    for y, y_vals in enumerate(x_vals):
        ax2.scatter(x+4, y+2, c = get_colour(y_vals), marker = 's', s = 250)

ax2.set_title('No Usable Ace', loc='left', fontsize=15, weight='bold')
ax2.set_xticks(np.arange(4,22,1))
ax2.set_yticks(np.arange(2,12,1))

plt.show()
# plt.savefig('images/learnt_grid.png')
