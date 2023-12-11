##
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# import cmasher as cmr

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('lines', linewidth=5)
page_width = 6.32
wd = "./visualization/"

def parse_strings(strings):
    tab_splited = strings.split("\t")
    mean_and_std = ([x.split(" ") for x in tab_splited])
    means = []
    stds = []
    for x in mean_and_std:
        means.append(float(x[0]))
        std = float(x[1].replace("(","").replace(")",""))
        stds.append(std)
    return np.array(means), np.array(stds)

##
################
# preliminary: SNR
################
N = 4
cm = plt.cm.viridis
colors = [cm(x) for x in np.linspace(0,1,N)]
ratio = 0.5
width = page_width * ratio
height = width / 2

snr = [3.98, 4.02, 4.08, 4.23][::-1]
ber = np.array([.986, .951, .900, .871])

plot_args = {'linestyle': 'solid', 'linewidth':1, "marker":"."}
fig, ax = plt.subplots()
ax.plot(snr, **plot_args, color=colors[0])
# color = line[0].get_color()
ax.set_ylim(3.9, 4.3)
labels = ['8b\n125T', '16b\n250T', '32b\n500T', '64b\n1000T']
ax.set_xticks(range(len(snr)), labels=labels)
ax_leg = ax.set_ylabel("SNR")
ax_leg.set_color(colors[0])

ax.set_title("Fixed BPT=.064")
ax2 = ax.twinx()
ax2.plot(ber, **plot_args, color=colors[1])
ax2.set_yticks(np.arange(0.8, 1, 0.05))
ax2.set_ylim(0.8, 1)
ax2_leg = ax2.set_ylabel("Bit Acc.")
ax2_leg.set_color(colors[1])

ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.8, top=0.8, wspace=0.2, hspace=0.2)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./snr.pdf")


##
import matplotlib.lines as mlines
################
#simple position scheme 1
################
ratio = 0.5
width = page_width * ratio
height = width / 2

clean_acc = {'vanilla':[.986, .951, .900, .871],
             'pos': [.927, .909, .901, .884]}

corrupted_acc = {'vanilla': [.974, .915, .861, .827],
                 'pos': [.614, .547, .54, .514]}

plot_args = {'linestyle': 'solid', 'linewidth':1}

fig, ax = plt.subplots()
# ax.scatter(acc, snr)
ax.plot(clean_acc['vanilla'], color='red', marker="o", **plot_args)
ax.plot(clean_acc['pos'], color='blue', marker="^", **plot_args)

plot_args = {'linestyle': 'dashed', 'linewidth':1,}
ax.plot(corrupted_acc['vanilla'], color='red', alpha=0.5, marker="o", **plot_args)
ax.plot(corrupted_acc['pos'], color='blue', alpha=0.5, marker="^", **plot_args)

ax.set_ylim(0.5, 1)
labels = ['8b', '16b', '24b', '32b']
ax.set_xticks(range(len(labels)), labels=labels)
ax_leg = ax.set_ylabel("Bit Acc.")

solid_line = mlines.Line2D([], [], color="black", label="clean", linewidth=0.5)
dashed_line = mlines.Line2D([], [], color="black", label="corrupted", linestyle='dashed', linewidth=0.5)
circle_marker = mlines.Line2D([], [], color='red', label='hash', linestyle='None', marker='o', markersize=5)
trig_marker = mlines.Line2D([], [], color='blue', label='cycle', linestyle='None', marker='^', markersize=5)

plt.legend(bbox_to_anchor=(1.0, 0.8), handles=[solid_line, dashed_line, circle_marker, trig_marker], prop={'size':6})

ax.set_title("Position Allocation Scheme")
ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.8, top=0.8, wspace=0.2, hspace=0.2)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./position.pdf")

##
import matplotlib.lines as mlines
################
#simple position scheme 2: only difference
################
ratio = 0.5
width = page_width * ratio
height = width / 2

clean_acc = {'vanilla':np.array([.986, .951, .900, .871]),
             'pos': np.array([.927, .909, .901, .884])}

corrupted_acc = {'vanilla': np.array([.974, .915, .861, .827]),
                 'pos': np.array([.614, .547, .54, .514])}

plot_args = {'linestyle': 'solid', 'linewidth':1}

fig, ax = plt.subplots()
# ax.scatter(acc, snr)
ax.plot(-clean_acc['vanilla'] + corrupted_acc['vanilla'], color='red', marker="o", **plot_args)
ax.plot(-clean_acc['pos'] + corrupted_acc['pos'], color='blue', marker="^", **plot_args)

# plot_args = {'linestyle': 'dashed', 'linewidth':1,}
# ax.plot(corrupted_acc['vanilla'], color='red', alpha=0.5, marker="o", **plot_args)
# ax.plot(corrupted_acc['pos'], color='blue', alpha=0.5, marker="^", **plot_args)

ax.set_ylim(-0.4, 0)
ax.set_yticks(ticks=[0, -0.1, -0.2, -0.3, -0.4], labels=["0", "-.1", "-.2", "-.3", "-.4"])
labels = ['8b', '16b', '24b', '32b']
ax.set_xticks(range(len(labels)), labels=labels)
ax_leg = ax.set_ylabel("Robustness")

circle_marker = mlines.Line2D([], [], color='red', label='pseudo-random', linestyle='None', marker='o', markersize=5)
trig_marker = mlines.Line2D([], [], color='blue', label='deterministic', linestyle='None', marker='^', markersize=5)

plt.legend(handles=[circle_marker, trig_marker], prop={'size':6})

ax.set_title("Position Allocation Schemes", size=10)
ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.8, top=0.8, wspace=0.2, hspace=0.2)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./position.pdf")

##
# acc. vs. quality @ delta

cm =  mpl.colormaps['Blues']
N = 6
colors = [cm(x) for x in np.linspace(0.8,1,N)]

ratio = 0.5
width = page_width * ratio
height = width / 2


psp_w_ref = [.379, .372, .371, .360, .336, .319]
psp_w_non_wm = [.460, .433, .417, .388, .349, .330]
ref_psp = 0.379
acc = [.766, .887, .947, .982, .993, .995]

ppl = np.array([4.64, 5.01, 5.6, 7.41, 10.3, 13.67])
ref_ppl = 4.39

plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
labels = [f"$\delta=${x}" for x in [1, 1.5, 2, 3, 4, 5]]
s = [10*(2**x) for x in range(N)]

ax.scatter(acc, psp_w_ref, marker=".", color=colors, s=s, alpha=0.7)
ax.axhline(ref_psp, linewidth=0.8, color="blue", alpha=0.5, linestyle="--")
# ax.set_xlim(0.6, 1.05)
# ax.set_ylim(0.2, 0.4)
ax.set_yticks(ticks=[0.325, 0.35, 0.375], labels=[".325", ".350", ".375"])
ax.set_xlabel("Bit Acc.")
lab = ax.set_ylabel("P-SP")
lab.set_color("blue")
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

cm =  mpl.colormaps['Reds']
colors = [cm(x) for x in np.linspace(0.8,1,N)]
ax2 = ax.twinx()
ax2.scatter(acc, ppl, s=s, color=colors, marker='.', alpha=0.7)
ax2.axhline(ref_ppl, linewidth=0.8, color="red", alpha=0.5, linestyle="dotted")
ax2.set_ylim(15, 3)

lab = ax2.set_ylabel("PPL")
lab.set_color('red')

ax.text(acc[0]-0.01, psp_w_ref[0]-0.02, labels[0])
ax.text(acc[1]-0.035, psp_w_ref[1]-0.015, labels[1])
ax.yaxis.set_minor_locator(AutoMinorLocator(3))
ax2.yaxis.set_minor_locator(AutoMinorLocator(3))

ax.grid(which="both")
plt.subplots_adjust(left=0.25, bottom=0.3, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("delta-vs-quality.pdf")

##
# quality across bit-width
ratio = 0.5
width = page_width * ratio
height = width / 2

fill_alpha = 0.3
fig, ax = plt.subplots()
psp_across_b = [0.498, 0.495, 0.501, 0.492, 0.498]
error_bar = np.array([.13, .13, .13, .13, .13]) / np.sqrt(500) * 3

x = np.arange(5)
# ax.errorbar(x=x, y=psp_across_b, yerr=error_bar, marker=".", linewidth=2,
#             linestyle="None", markersize=10, color="b", alpha=0.5)
ax.plot(x, psp_across_b, linewidth=2, marker=".", color="b")
ax.fill_between(x, psp_across_b+error_bar, psp_across_b-error_bar, alpha=fill_alpha, color="b")

labels = ["0b", '8b', '16b', '24b', '32b']
ax.set_xticks(range(len(labels)), labels=labels)
lab = ax.set_ylabel("P-SP")
lab.set_color("b")
ax.set_xlabel("Bit Width")
ax.set_ylim(0.4, 0.6)


ppl_mean, ppl_std = parse_strings("5.09 (1.6)\t5.18 (1.5)	5.15 (1.6)	5.14 (1.5)	5.21 (1.5)")
error_bar = ppl_std / np.sqrt(500) * 3
ax2 = ax.twinx()
x = np.arange(5)
# ax2.errorbar(x=x, y=ppl_mean, yerr=error_bar, marker=".", linewidth=2,
#              linestyle="None", markersize=10, color="r", alpha=0.5)

ax2.plot(x, ppl_mean, linewidth=2, marker=".", color="r")
ax2.fill_between(x, ppl_mean+error_bar, ppl_mean-error_bar, alpha=fill_alpha, color="r")

lab = ax2.set_ylabel("PPL")
lab.set_color("r")
ax.yaxis.set_minor_locator(AutoMinorLocator(3))
ax2.yaxis.set_minor_locator(AutoMinorLocator(3))
ax2.yaxis.set_label_coords(1.25, 0.1)

ax3 = ax.twinx()
latency = [8.2, 8.0, 8.0, 8.0, 8.2]
errorbar = [.1, .1, .1, .1, .1] / np.sqrt(500) * 3
# x = x + 0.15
# ax3.errorbar(x=x, y=latency, yerr=error_bar, marker=".", linewidth=2,
#              linestyle="None", markersize=10, color="grey", alpha=0.5)
ax2.plot(x, latency, linewidth=2, marker=".", color="grey")
ax2.fill_between(x, latency+error_bar, latency-error_bar, alpha=fill_alpha, color="grey")
lab = ax3.set_ylabel("Latency")
ax3.yaxis.set_label_coords(1.25, 0.7)
ax3.yaxis.set_ticklabels([])
# ax3.xaxis.set_ticklabels([])
# ax3.axis("off")
lab.set_color("grey")
ax2.set_ylim(4.5, 9)


ax.grid()
plt.subplots_adjust(left=0.25, bottom=0.3, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("quality-vs-bit.pdf")

##
# acc. vs. quality @ delta
cm =  mpl.colormaps['Blues']

ratio = 0.5
width = page_width * ratio
height = width / 2

psp_w_ref = [0.384, 0.385, 0.376, 0.356, 0.325, 0.244]
acc = [0.618, 0.751, 0.917, 0.979, 0.997, 0.994]

plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()

ax.scatter(acc, psp_w_ref, marker=".", color="k", alpha=0.5)

psp_data_dict = {'delta=0.5': [0.384],
                 'delta=1': [0.382, 0.380],
                 'delta=2': [.371, .365]
                 }

acc_data_dict = {'delta=0.5': [0.831],
                 'delta=1': [0.864, 0.895],
                 'delta=2': [.955, .962]
                 }

ax.scatter(acc_data_dict['delta=1'], psp_data_dict['delta=1'], marker=".", color="b", alpha=0.5)
ax.scatter(acc_data_dict['delta=2'], psp_data_dict['delta=2'], marker=".", color="r", alpha=0.5)





# ax.set_ylim(0.2, 0.4)
# ax.set_xlabel("Bit Acc.")
# lab = ax.set_ylabel("P-SP")
# lab.set_color("blue")

ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("delta-vs-quality.pdf")

## main: clean
ratio = 1
# page_width = 5
width = page_width * ratio
height = width / 3.2

main_mean, main_std = parse_strings(".986 (0.06)	.951 (.07)	.900 (.09)	.871 (0.08)")
r2_mean, r2_std = parse_strings(".966 (.07)	.905 (.08)	.858 (.08)	0.820 (.08)")
r2g2_mean, r2g2_std = parse_strings(".978 (.05)	.922 (.07)	.875 (.08)	0.849 (.07)")
data_mean = [main_mean, r2_mean, r2g2_mean]
data_std = [main_std, r2_std, r2g2_std]
labels = ["$\gamma=0.25,r=4$", "$\gamma=0.25,r=2$", "$\gamma=0.5,r=2$"]
list_decoded_mean = [.986, .951, .900, .871]

fig, ax = plt.subplots(1, 2)

N = 4
x = np.arange(N)

bar_width = 0.25
multiplier = 0

num_groups = 3
cm = plt.cm.viridis
cm = plt.cm.inferno

colors = [cm(x) for x in np.linspace(0, 0.7, num_groups)]
hatches = ['', '.', '-']

for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax[0].bar(x + offset, data_mean[idx], bar_width, yerr=data_std[idx] / np.sqrt(500) * 3,
                   label=labels[idx], color=colors[idx], error_kw={'elinewidth': 1}, capsize=2, hatch=hatches[idx],
                      alpha=0.8
                      )
    multiplier += 1

multiplier = 0
offset = bar_width * multiplier
# for ix in range(1, len(x)):
#     ax.text(x[ix] - 0.07, data_mean[0][ix] + 0.01, "*")

rects = ax[0].bar(x + offset, list_decoded_mean, bar_width, color="None", edgecolor=colors[0],
               label="16-List decoded", alpha=0.5)

ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))
# ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax[0].set_ylim(0.8, 1)
ax[0].set_ylabel("Bit Acc.")
ax[0].grid(axis='y')
ax[0].set_xticks(x + bar_width, ["8b", "16b", "24b", "32b"])
plt.text(0.25, 0.025, "250T",  transform=plt.gcf().transFigure)
ax[0].legend(prop={'size': 6}, ncols=4, bbox_to_anchor=(1.1, 1.2), loc="center", handleheight=2,
             handlelength=3)


acc_mean, acc_std = parse_strings(".846 (.09)	0.913 (.08)	.951 (.07)	.958 (.09)")
acc_mean = acc_mean[::-1]
acc_std = acc_std[::-1]
x = np.arange(len(acc_mean))
ax[1].bar(x, acc_mean, bar_width, color=colors[0], alpha=0.8)

list_decoded_mean, _ = parse_strings(".876 (.08)	.954 (.06)	.999 (.03)	.992 (.04)")
list_decoded_mean = list_decoded_mean[::-1]

# print((acc_mean - list_decoded_mean).mean())
ax[1].bar(x, list_decoded_mean, bar_width, color="None", edgecolor=colors[0], linewidth=0.75,
          alpha=0.7)

x_labels = ['8b\n125T', '16b\n250T', '32b\n500T', '64b\n1000T']
ax[1].set_xticks(ticks=x, labels=x_labels)
ax[1].set_ylim(0.5, 1)
ax[0].set_ylim(0.5, 1)

for ax_ in ax:
    ax_.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_.grid(which="major", linewidth=0.3, axis="y")
    ax_.grid(which="minor", linewidth=0.25, linestyle="--", axis="y",)
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.8, wspace=.2, hspace=0.)
plt.show()
fig.set_size_inches(width, height, forward=True)
fig.savefig("./visualization/main-clean.pdf")


## main: comparison with others
ratio = 0.8
width = page_width * ratio
height = width / 2.5
bar_plot_args = {'error_kw':{'elinewidth': 0.5}, 'capsize': 1}


main_mean, main_std = parse_strings(".986 (0.06)	.981 (.07)	.956 (.10)	.900 (.13)")
cs_green_mean, cs_green_std = parse_strings(".995 (.05)	.988 (.08)	.970 (.12)	.908 (.20)")
cs_ems_mean, cs_ems_std = parse_strings(".979 (.10)	.943 (.17)	.858 (.24)	.800 (.28)")
msg_hash_mean, msg_hash_std = parse_strings(".977 (.11)	.973 (.12)	.951 (.16)	.858 (.24)")

data_mean = [main_mean, cs_green_mean, cs_ems_mean, msg_hash_mean]
data_std = [main_std, cs_green_std, cs_ems_std, msg_hash_std]
labels = ["MPAC", "CS (Greenlist)", "CS (EMS)", "MSG-HASH"]

fig, ax = plt.subplots(1, 3)

N = 4
x = np.arange(N)

bar_width = 0.15
multiplier = 0

num_groups = 4
cm = plt.cm.viridis
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]

for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax[0].bar(x + offset, data_mean[idx], bar_width, yerr=data_std[idx] / np.sqrt(500) * 3,
                   label=labels[idx], color=colors[idx], **bar_plot_args)
    multiplier += 1

multiplier = 0
offset = bar_width * multiplier
# for ix in range(1, len(x)):
#     ax[0].text(x[ix] - 0.07, data_mean[0][ix] + 0.01, "*")

# second plot
main_mean, main_std = parse_strings(".951 (.07)	.939 (.08)	.887 (.09)	.819 (.12)")
cs_green_mean, cs_green_std = parse_strings(".01 (.01)	.01 (.01)	.01 (.01)	.01 (.01)")
cs_ems_mean, cs_ems_std = parse_strings(".905 (.20)	.811 (.26)	0.702 (.26)	.601 (.23)")
msg_hash_mean, msg_hash_std = parse_strings(".936 (.18)	.909 (.20)	.810 (.26)	.614 (.22)")

data_mean = [main_mean, cs_green_mean, cs_ems_mean, msg_hash_mean]
data_std = [main_std, cs_green_std, cs_ems_std, msg_hash_std]


N = 4
x = np.arange(N)

bar_width = 0.15
multiplier = 0

num_groups = 4
cm = plt.cm.viridis
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]

for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax[1].bar(x + offset, data_mean[idx], bar_width, yerr=data_std[idx] / np.sqrt(500) * 3,
                   label=labels[idx], color=colors[idx], **bar_plot_args)
    multiplier += 1

multiplier = 0

# third plot
main_mean, main_std = parse_strings(".899 (.09)	.882 (.09)	.830 (.10)	.755 (.11)")
cs_green_mean, cs_green_std = parse_strings(".01 (.01)	.01 (.01)	.01 (.01)	.01 (.01)")
cs_ems_mean, cs_ems_std = parse_strings(".775 (.26)	.729 (.24)	.633 (.23)	.513 (.13)")
msg_hash_mean, msg_hash_std = parse_strings(".876 (.22)	.828 (.25)	.663 (.26)	.516 (.16)")

data_mean = [main_mean, cs_green_mean, cs_ems_mean, msg_hash_mean]
data_std = [main_std, cs_green_std, cs_ems_std, msg_hash_std]


N = 4
x = np.arange(N)

bar_width = 0.15
multiplier = 0

num_groups = 4
cm = plt.cm.viridis
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]
num_samples = [500, 100, 500, 500]
for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax[2].bar(x + offset, data_mean[idx], bar_width, yerr=data_std[idx] / np.sqrt(num_samples[idx]) * 3,
                   label=labels[idx], color=colors[idx], **bar_plot_args)
    multiplier += 1

multiplier = 0




for ax_ in ax:
    ax_.yaxis.set_minor_locator(AutoMinorLocator(4))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_.set_ylim(0.5, 1)
    ax_.grid(which="major", linewidth=0.3, axis="y")
    ax_.grid(which="minor", linewidth=0.25, linestyle="--", axis="y",)

    ax_.set_xticks(x + bar_width, ["Clean", "10%", "30%", "50%"])

labels = [item.get_text() for item in ax[1].get_yticklabels()]

ax[1].set_yticklabels([" "]*len(labels))
ax[2].set_yticklabels([" "]*len(labels))
ax[0].set_xlabel("8b")
ax[1].set_xlabel("16b")
ax[2].set_xlabel("24b")
ax[0].legend(prop={'size': 7}, ncols=4, bbox_to_anchor=(0.2, 1.05), loc="lower left")
ax[0].set_ylabel("Bit Acc.")
plt.subplots_adjust(left=0.12, bottom=0.3, right=0.98, top=0.8, wspace=.05, hspace=0.)
fig.set_size_inches(width, height, forward=True)
plt.show()
fig.savefig("./visualization/main-comparison.pdf")

##
width = page_width * 0.2
# height = width * 2
# height = page_width * 0.3

# height = width / 1
fig, ax = plt.subplots(1,1)
cm = plt.cm.viridis
colors = [cm(x) for x in np.linspace(0.2, 1, 3)]

ours = [0.986, .974, 0.96, .951]
cs = [0.977, 0.957, 0.917]
msg_hash = [0.98, 0.9, 0.678]
x_axis = [1e-2, 1e-3, 1e-4, 1e-5]
x_axis = np.arange(len(ours))
labels = ['MPAC', "C-SHIFT", "MSG-HASH"]
for idx, data in enumerate([ours, cs, msg_hash]):
    ax.plot(x_axis[:len(data)], data, c=colors[idx], linewidth=2, label=labels[idx], marker=".")

ax.plot(x_axis[-2:], [.917, .877], marker=".", linestyle="dotted", linewidth=2, c=colors[1])
ax.plot(x_axis[-2:], [.678, .456], marker=".", linestyle="dotted", linewidth=2, c=colors[2])

ax.set_xticks(x_axis, ["1E-2", "1E-3", "1E-4", "1E-5"])
# ax.set_xscale('log')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_ylim(0.6, 1.0)

# ax.legend(prop={'size': 6}, ncols=1, loc="best")
plt.subplots_adjust(left=0.3, bottom=0.3, right=0.9, top=0.95)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./visualization/tpr-at-fpr.pdf")


## robustness : cp
ratio = 0.8
width = page_width * ratio
height = width / 2.4
data_strings = """
.986 (.06)	.981 (.07)	0.971 (.08)	.956 (.10)	.938 (.12)	.900 (.13)
.951 (.07)	.939 (.08)	.918 (.09)	.887 (.09)	.858 (.11)	.819 (.12)
.899 (.09)	.882 (.09)	.858 (.10)	.830 (.10)	.797 (.11)	.755 (.11)
.871 (.08)	.851 (.09)	.828 (.09)	.801 (.09)	.765 (.09)	.723 (.1)
"""
acc_by_bits = []
for row in data_strings.strip().split("\n"):
    acc_mean, acc_std = parse_strings(row.strip())
    acc_mean = acc_mean.reshape(1, -1)
    acc_by_bits.append(acc_mean)

acc_by_groups = np.concatenate(acc_by_bits, axis=0)

data_strings = """
.997 (.02)	.997 (.02)	.995 (.03)	.993 (.03)	.991 (.04)	.980 (.05)
.988 (0.04)	.983 (.04)	.978 (.05)	.964 (.06)	.947 (.07)	.918 (.08)
.959 (.06)	.944 (.06)	.927 (.08)	.907 (.08)	.879 (.09)	.840 (.09)
.927 (.07)	.910 (.08)	.888 (.08)	.863 (.08)	.831 (.09)	.792 (.09)
"""
list_acc_by_bits = []
for row in data_strings.strip().split("\n"):
    acc_mean, acc_std = parse_strings(row.strip())
    acc_mean = acc_mean.reshape(1, -1)
    list_acc_by_bits.append(acc_mean)

list_acc_by_groups = np.concatenate(list_acc_by_bits, axis=0)

labels = ["Clean", "cp=0.1", "cp=0.2", "cp=0.3", "cp=0.4", "cp=0.5"]
# b8_acc, b8_std = parse_strings(".986 (.06)	.981 (.07)	0.971 (.08)	.956 (.10)	.938 (.12)	.900 (.13)")
# b8_list_acc, b8_list_std = parse_strings(".997 (.02)	.997 (.02)	.995 (.03)	.993 (.03)	.991 (.04)	.980 (.05)")

fig, ax = plt.subplots()

N = 4
x = np.arange(N)

bar_width = 1/8
multiplier = 0

num_groups = 6
cm = plt.cm.inferno
colors = [cm(x) for x in np.linspace(0, 0.8, num_groups)]

for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax.bar(x + offset, acc_by_groups[:, idx], bar_width, color=colors[idx], error_kw={'elinewidth': 1},
                   label=labels[idx])
    rects = ax.bar(x + offset, list_acc_by_groups[:, idx], bar_width, color="None", edgecolor=colors[idx],
                   linewidth=0.75)
    multiplier += 1


ax.yaxis.set_minor_locator(AutoMinorLocator(4))
# ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.set_ylim(0.5, 1)
ax.set_ylabel("Bit Acc.")
ax.grid(axis='y')
ax.grid(which="major", linewidth=0.5, axis="y")
ax.grid(which="minor", linewidth=0.25, linestyle="--", axis="y",)
ax.set_xticks(x + num_groups * bar_width / 2, ["8b", "16b", "24b", "32b"])
plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.85, wspace=0., hspace=0.)
ax.legend(prop={'size': 6}, ncols=num_groups, bbox_to_anchor=(-0.1, 1.05), loc="lower left")
plt.show()
fig.set_size_inches(width, height, forward=True)
fig.savefig("./robustness.pdf")

## gpt-robustness
width = page_width * 0.2
# height = width / 2

bar_width = 0.3
acc_mean, acc_std = parse_strings("0.733 (.19)	.792 (.19)	.795 (.19)")
list_mean, list_std = parse_strings("0.911 (.10)	.934 (.09)	.939 (.09)")
xlabels = ["250T", "400T", "500T"]
x = np.arange(0, bar_width * 6, bar_width * 2)
fig, ax = plt.subplots()
ax.bar(x, acc_mean, bar_width, color=colors[1], label="GPT-3.5")
ax.bar(x, list_mean, bar_width, color=colors[1], edgecolor=colors[1], linewidth=0.75, alpha=0.1)

list_mean, list_std = parse_strings(".893 (.12)	.924 (.11)	.928 (.11)")
ax.bar(x, list_mean, bar_width, color=colors[1], edgecolor=colors[1], linewidth=0.75, alpha=0.2)

list_mean, list_std = parse_strings(".856 (.14)	.894 (.13)	.898 (.13)")
ax.bar(x, list_mean, bar_width, color=colors[1], edgecolor=colors[1], linewidth=0.75, alpha=0.25)

list_mean, list_std = parse_strings(".825 (.16)	.874 (.15)	.875 (.15)")
ax.bar(x, list_mean, bar_width, color=colors[1], edgecolor=colors[1], linewidth=0.75, alpha=0.3)


ax.set_xticks(x, xlabels, fontsize=8)
ax.set_yticklabels([])
ax.legend(prop={'size': 6}, bbox_to_anchor=(0.0, 1.), loc="lower left")
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85, wspace=0., hspace=0.)


ax.set_ylim(0.5, 1)
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

ax.grid(axis='y')
ax.grid(which="major", linewidth=0.5, axis="y")
ax.grid(which="minor", linewidth=0.25, linestyle="--", axis="y",)
fig.set_size_inches(width, height, forward=True)

plt.show()
fig.savefig("./robustness-gpt.pdf")


## discussion: gamma-radix
ratio = 0.5
page_width = 5
width = page_width * ratio
height = width / 1.6

cm = plt.cm.inferno
num_groups = 3
colors = [cm(x) for x in np.linspace(0, 0.7, num_groups)]

acc_mean, acc_std = parse_strings(".840 (.14)	.904 (.14)	.891 (.17)	.897 (.12)	.949 (.11)	.923 (.11)")
acc_std = acc_std / np.sqrt(1000) * 3
auc = [0.991, 0.982, 0.968, 0.991, 0.985, 0.98]

fig, ax = plt.subplots()
plot_kwargs = {'linewidth': 0.5, 'markersize': 2}
cnt = 0
ax.plot(acc_mean[cnt:cnt+3], auc[cnt:cnt+3], "-o", **plot_kwargs, label=".125", color=colors[0])
cnt = 3
ax.plot(acc_mean[cnt:cnt+2], auc[cnt:cnt+2], "-o", **plot_kwargs, label=".25", color=colors[1])
cnt = 5
ax.plot(acc_mean[cnt:cnt+1], auc[cnt:cnt+1], "-o", **plot_kwargs, label=".5", color=colors[2])
ax.set_xlim(0.8, 0.97)
ax.set_ylim(0.96, 1)
ax.set_xlabel("Bit Acc.")
ax.set_ylabel("AUC")
legend = ax.legend(prop={'size':6})
legend.get_title().set_fontsize('5')
plt.subplots_adjust(left=0.25, bottom=0.3, right=0.95, top=0.95, wspace=0., hspace=0.)
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))

colors = [colors[0], colors[0], colors[0], colors[1], colors[1], colors[2]]
for c_idx in range(6):
    ax.plot([acc_mean[c_idx] - acc_std[c_idx], acc_mean[c_idx] + acc_std[c_idx]], [auc[c_idx], auc[c_idx]], "--|",
            markersize=4, linewidth=0.5, color=colors[c_idx])

idx = 0
labels = ["r=2", "4", "8", "r=2", "4", "r=2"]
for x, y, l in zip(acc_mean, auc, labels):
    if idx == 5:
        ax.text(x + 0.00, y-0.005, l, size=8)
    else:
        ax.text(x + 0.004, y+0.001, l, size=8)
    idx += 1
ax.grid()
plt.show()
fig.set_size_inches(width, height, forward=True)
# fig.savefig("./gamma-radix.pdf")

## auc@T across bit-width
ratio = 0.5
page_width = 5
width = page_width * ratio
height = width / 1.6

import pandas as pd
df = pd.read_csv("visualization/data/auc_at_t_bit_width.csv")[['name', 'idx_T', 'aucs']]
cm = plt.cm.YlGnBu
num_groups = 5
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]

plot_kwargs = {'linewidth': 1}
fig, ax = plt.subplots()
idx = 0
auc_at_50 = []
auc_at_100 = []
auc_at_200 = []

for col in df['name'].unique():
    data = df.loc[df['name'] == col]
    ax.plot(data['idx_T'], data['aucs'], color=colors[idx] ,**plot_kwargs)
    auc_at_50.append(data['aucs'].iloc[49])
    auc_at_100.append(data['aucs'].iloc[99])
    auc_at_200.append(data['aucs'].iloc[199])
    idx += 1

arrowprops = dict(facecolor='grey', shrink=0.05, width=0.1, headwidth=3, headlength=2)

ax.annotate(f"AUC={auc_at_50[-1]:.2f}", xy=(50, auc_at_50[-1]),
            xytext=(50, 0.6),
            arrowprops=arrowprops)

ax.annotate(f"AUC={auc_at_100[-1]:.2f}", xy=(100, auc_at_100[-1]),
            xytext=(100, 0.7),
            arrowprops=arrowprops)

ax.annotate(f"AUC={auc_at_200[-1]:.2f}", xy=(200, auc_at_200[-1]),
            xytext=(160, 0.8),
            arrowprops=arrowprops)

ax.set_xlabel("# of Tokens Observed")
ax.set_ylabel("AUC")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))

ax.grid(which="both")
plt.show()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.95, top=0.95, wspace=0., hspace=0.)
fig.set_size_inches(width, height, forward=True)
fig.savefig("visualization/auc-at-t.pdf")


## z-score curve w.r.t observed tokens
ratio = 1.0
page_width = 5
width = page_width * ratio
height = width / 2

import pandas as pd
df = pd.read_csv("./data/no_wm_z_score_at_t.csv")[['avg', 'idx_T', 'name']]
df = df[df['name'] != 'fixedT-1b-250T-2R-0.25GAMMA-lefthash_eval']
print(df['name'].unique())
cm = plt.cm.YlGnBu
num_groups = 5
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]

plot_kwargs = {'linewidth': 1}
fig, ax = plt.subplots(1, 3)
idx = 0
labels = ['8b', '16b', '24b', '32b']

no_wm_z = []
for col in df['name'].unique():
    data = df.loc[df['name'] == col].iloc[:-1]
    no_wm_z.append(data)
    ax[0].plot(data['idx_T'], data['avg'], label=labels[idx], color=colors[idx], **plot_kwargs)
    idx += 1





ax[0].legend(prop={'size': 8}, ncol=4, loc="lower center", bbox_to_anchor=(1.65, 1.15))

df = pd.read_csv("./data/w_wm_z_score_at_t.csv")[['avg', 'idx_T', 'name']]
print(df['name'].unique())
idx = 0
wm_z = []

for col in df['name'].unique():
    data = df.loc[df['name'] == col].iloc[:-2]
    wm_z.append(data)
    ax[1].plot(data['idx_T'], data['avg'], label=labels[idx], color=colors[idx], **plot_kwargs)
    idx += 1

ax[1].legend()
ax[0].set_ylabel("Z-score")
ax[0].set_title("No watermark")
ax[1].set_title("Watermarked")
ax[2].set_title("Difference")

for ax_ in ax:
    ax_.grid(which="both")
    ax_.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_.xaxis.set_minor_locator(AutoMinorLocator(4))


idx = 0
for w_wm, no_wm in zip(wm_z, no_wm_z):
    ax[2].plot(range(len(w_wm)), w_wm['avg'].values - no_wm['avg'].values, color=colors[idx], **plot_kwargs)
    idx += 1

fig.text(0.35, 0.05, "# of Tokens Observed")
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.8, wspace=0.3, hspace=0.)
plt.show()
fig.set_size_inches(width, height, forward=True)
fig.savefig("./z-at-t.pdf")


## auc at t w.r.t radix

ratio = 0.5
page_width = 5
width = page_width * ratio
height = width / 1.6

import pandas as pd
df = pd.read_csv("./data/no_wm_z_score_at_t_radix.csv")[['name', 'idx_T', 'avg']]
print(df.name.unique())

labels = ['2', '4', '8']
cm = plt.cm.YlGnBu
num_groups = 5
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]

plot_kwargs = {'linewidth': 1}
fig, ax = plt.subplots()
idx = 0
for col in df['name'].unique():
    data = df.loc[df['name'] == col].iloc[:-2]
    ax.plot(data['idx_T'], data['avg'], color=colors[idx], label=labels[idx], **plot_kwargs)
    idx += 1


ax.legend(prop={'size': 5}, title="Radix", ncol=3, loc="lower center", bbox_to_anchor=(.5, 1))
ax.set_xlabel("# of Tokens Observed")
ax.set_ylabel("Z-score")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))

ax.grid(which="both")
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.8, wspace=0., hspace=0.)
fig.set_size_inches(width, height, forward=True)
plt.show()

fig.savefig("./z-score-radix.pdf")


## appendix: simulation

import numpy as np
num_trials = 1000
T = 500
p = 16
r = 4


agg = lambda x: np.sum(x)
gamma = 0.25
diff = []

for p in [8, 16, 24, 32]:
    for r in [4]:
        print(f"Num positions={p}, Radix={r}")
        prob = [gamma for _ in range(r)]
        prob = prob / np.sum(prob)
        mu = T * gamma
        sigma = np.sqrt(T * prob[0] * (1-prob[0]))
        data = []
        max_w_data = []

        for _ in range(num_trials):
            aggregate_w = []
            for _ in range(p):
                max_w = np.random.multinomial(T // p, prob)[:r].max()
                aggregate_w.append(max_w)
            max_w_data.append(agg(aggregate_w))
            statistic = np.sum(aggregate_w)
            z = (statistic - mu) / sigma
            data.append(z)


        non_wm_score = data
        print(f"\tMean statistics ={np.mean(non_wm_score)} std={np.std(max_w_data)}")
        # print(np.mean(data))
        prob = [gamma for _ in range(r)]
        prob[0] += 0.3
        prob = prob / np.sum(prob)
        mu = T * gamma
        sigma = np.sqrt(T * (gamma) * (1 - gamma))
        data = []
        max_w_data = []

        for _ in range(num_trials):
            aggregate_w = []
            for _ in range(p):
                max_w = np.random.multinomial(T // p, prob)[:r].max()
                aggregate_w.append(max_w)

            max_w_data.append(agg(aggregate_w))
            statistic = agg(aggregate_w)
            z = (statistic - mu) / sigma
            data.append(z)

        wm_score = data
        print(f"    Mean statistics ={np.mean(wm_score)} std={np.std(max_w_data)}")
        print(f"Diff={(np.mean(wm_score) - np.mean(non_wm_score))}")
        diff.append((np.mean(wm_score) - np.mean(non_wm_score)))

error_bar = np.sqrt(np.std(wm_score)**2 + np.std(non_wm_score)**2) / np.sqrt(num_trials) * 3


ratio = 0.5
page_width = 5
width = page_width * ratio
height = width / 1.6

fig, ax = plt.subplots()

x = np.arange(4)
ax.plot(x, diff, linewidth=2, marker=".")
ax.fill_between(x, diff-error_bar, diff+error_bar, alpha=fill_alpha, color="b")
ax.set_xticks(x, ["8b", "16b", "24b", "32b"])
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.set_ylabel("Score $\Delta$")
ax.set_xlabel("Bit-Width")
ax.grid()
plt.show()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.8, wspace=0., hspace=0.)
fig.set_size_inches(width, height, forward=True)
fig.savefig("./simulation-bit.pdf")


## appendix: simulation

import numpy as np
num_trials = 1000
T = 500
p = 16
r = 4

agg = lambda x: np.sum(x)
gamma = 0.125
diff = []
group = [2, 4, 8]
for p in [8]:
    for r in group:
        print(f"Num positions={p}, Radix={r}")
        prob = [gamma for _ in range(r)]
        prob = prob / np.sum(prob)
        mu = T * gamma
        sigma = np.sqrt(T * prob[0] * (1-prob[0]))
        data = []
        max_w_data = []

        for _ in range(num_trials):
            aggregate_w = []
            for _ in range(p):
                max_w = np.random.multinomial(T // p, prob)[:r].max()
                aggregate_w.append(max_w)
            max_w_data.append(agg(aggregate_w))
            statistic = np.sum(aggregate_w)
            z = (statistic - mu) / sigma
            data.append(z)


        non_wm_score = data
        print(f"\tMean statistics ={np.mean(non_wm_score)} std={np.std(max_w_data)}")
        # print(np.mean(data))
        prob = [gamma for _ in range(r)]
        prob[0] += 0.3
        prob = prob / np.sum(prob)
        mu = T * gamma
        sigma = np.sqrt(T * (gamma) * (1 - gamma))
        data = []
        max_w_data = []

        for _ in range(num_trials):
            aggregate_w = []
            for _ in range(p):
                max_w = np.random.multinomial(T // p, prob)[:r].max()
                aggregate_w.append(max_w)

            max_w_data.append(agg(aggregate_w))
            statistic = agg(aggregate_w)
            z = (statistic - mu) / sigma
            data.append(z)

        wm_score = data
        print(f"    Mean statistics ={np.mean(wm_score)} std={np.std(max_w_data)}")
        print(f"Diff={(np.mean(wm_score) - np.mean(non_wm_score))}")
        diff.append((np.mean(wm_score) - np.mean(non_wm_score)))

error_bar = np.sqrt(np.std(wm_score)**2 + np.std(non_wm_score)**2) / np.sqrt(num_trials) * 3

ratio = 0.5
page_width = 5
width = page_width * ratio
height = width / 1.6

fig, ax = plt.subplots()

x = np.arange(len(group))
ax.plot(x, diff, linewidth=2, marker=".")
ax.fill_between(x, diff-error_bar, diff+error_bar, alpha=fill_alpha, color="b")
ax.set_xticks(x, group)
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.set_xlabel("Radix")
ax.set_ylabel("Score $\Delta$")
ax.grid()
plt.show()
plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.8, wspace=0., hspace=0.)
fig.set_size_inches(width, height, forward=True)
fig.savefig("./simulation-radix.pdf")

## model ablations
ratio = 0.9
width = page_width * ratio
height = width / 2
data_strings = """
.986 (.06)	.733 (.20)	.960 (.10)	.977 (.07)
.994 (.03)	.691 (.18)	.963 (.10)	.971 (.09)
.978 (.08)	.624 (.19)	.935 (.11)	.963 (.09)
"""
mean_data = []
std_data = []
for row in data_strings.strip().split("\n"):
    acc_mean, acc_std = parse_strings(row.strip())
    acc_mean = acc_mean.reshape(1, -1)
    acc_std = acc_std.reshape(1, -1)
    mean_data.append(acc_mean)
    std_data.append(acc_std)

acc_by_groups = np.concatenate(mean_data, axis=0)
std_by_groups = np.concatenate(std_data, axis=0)
std_by_groups = std_by_groups / np.sqrt(500) * 3
# std_by_groups[1:, :] /= np.sqrt(100) * 3

labels = ["7b", "13b", "70b"]

fig, ax = plt.subplots()

N = 4
x = np.arange(N)

bar_width = 1/4
multiplier = 0

num_groups = 3
cm = plt.cm.GnBu
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]
for idx in range(num_groups):
    offset = bar_width * multiplier
    rects = ax.bar(x + offset, acc_by_groups[idx, :].flatten(), bar_width, yerr=std_by_groups[idx, :].flatten(),
                        color=colors[idx], label=labels[idx], error_kw={'elinewidth': 1}, capsize=2)
    multiplier += 1


ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.set_ylim(0.5, 1)
ax.set_ylabel("Bit Acc.")
ax.grid(axis='y')
ax.grid(which="major", linewidth=0.5, axis="y")
ax.grid(which="minor", linewidth=0.25, linestyle="--", axis="y",)
ax.set_xticks(x + bar_width, ['C4\n(Newslike)', 'LFQA', "Essays", "Wikitext"])
plt.subplots_adjust(left=0.12, bottom=0.2, right=0.95, top=0.85, wspace=0., hspace=0.)
ax.legend(prop={'size': 8}, ncols=num_groups, bbox_to_anchor=(0.15, 1.0), loc="lower left")
plt.show()
fig.set_size_inches(width, height, forward=True)
fig.savefig(os.path.join(wd,"./model-size.pdf"))


## feedback

cm =  mpl.colormaps['Blues']
N = 6
colors = [cm(x) for x in np.linspace(0.8,1,N)]

ratio = 0.5
width = page_width * ratio
height = width / 2


psp_w_ref = [.379, .372, .371, .360, .336, .319]
acc = [.766, .887, .947, .982, .993, .995]

feedback_acc = [.769, .822, .860, .874, .891, .901, .957, .960]
feedback_psp = [.390, .376, .362, .375, .372, .36, .364, .366]


plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots()
labels = [f"$\delta=${x}" for x in [1, 1.5, 2, 3, 4, 5]]
# s = [10*(2**x) for x in range(N)]

ax.scatter(acc, psp_w_ref, marker=".", color="b", alpha=0.7)
ax.scatter(feedback_acc, feedback_psp, marker=".", color="r", alpha=0.7)

ax.axhline(ref_psp, linewidth=0.8, color="blue", alpha=0.5, linestyle="--")
# ax.set_xlim(0.6, 1.05)
ax.set_ylim(0.2, 0.4)
# ax.set_yticks(ticks=[0.325, 0.35, 0.375], labels=[".325", ".350", ".375"])
ax.set_xlabel("Bit Acc.")
lab = ax.set_ylabel("P-SP")
lab.set_color("blue")
# ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# ax.yaxis.set_minor_locator(AutoMinorLocator(3))

ax.grid(which="both")
plt.subplots_adjust(left=0.25, bottom=0.3, right=0.8, top=0.95, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.show()
# fig.savefig("feedback-delta.pdf")


## appendix: confidence vs. error rate
from watermark_reliability_release.utils.io import read_jsonlines
import numpy as np
import os
filename = "./watermark_reliability_release/experiments/llama-7b/fixedT-8b-250T-4R-0.25GAMMA-lefthash_eval/gen_table_w_metrics.jsonl"
raw_data = [x for x in read_jsonlines(filename)]
ratio = 0.5
width = page_width * ratio
height = width / 1.6

bw = 8
total_bits = list(range(bw))
confidence = [np.array(x['w_wm_output_confidence_per_position']) for x in raw_data]
filter_idx = [idx for idx, c in enumerate(confidence) if len(c) == 4]
confidence = [c for c in confidence if len(c) == 4]
confidence = [np.repeat(x, 2) for x in confidence]


error_pos = [x['w_wm_output_error_pos'] for x in raw_data]
error_pos = [error_pos[idx] for idx in filter_idx]


error_per_pos = []
for err in error_pos:
    row = np.zeros(bw)
    row[err] = 1
    error_per_pos.append(row)


confidence = np.concatenate(confidence, axis=0).flatten()
confidence = 1 - confidence
error_per_pos = np.concatenate(error_per_pos, axis=0).flatten()

num_bins = 10
val, edges = np.histogram(confidence, num_bins)
bin_idx = np.digitize(confidence, edges[:-1])
mean_error = []
for i in np.unique(bin_idx):
    mean_error.append(error_per_pos[bin_idx == i].mean())




fig, ax = plt.subplots()
x = np.arange(0.05, 1, 1 / num_bins)
ax.bar(x, mean_error, 0.1)
# ax.plot([0, 1], [0, 1], "--", linewidth=1)

ticks = [f".{x}" for x in range(10)]
ax.set_xticks(np.arange(0, 1, .1), ticks)
ax.set_xlim(0, 1)
# ax.xaxis.set_major_locator(MultipleLocator(.1))
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.set_xlabel("Confidence")
ax.set_ylabel("Error Rate")
ax.grid(which="both")
plt.subplots_adjust(left=0.2, bottom=0.25, right=0.9, top=0.95, wspace=0.1, hspace=0.1)
fig.set_size_inches(width, height)
plt.show()
fig.savefig("./confidence-vs-error.pdf")


## auc@T across bit-width
ratio = 0.8
page_width = 5
width = page_width * ratio
height = width / 1.6

import pandas as pd
df = pd.read_csv("visualization/data/cp-auc-at-t.csv")[['name', 'idx_T', 'aucs']]
cm = plt.cm.YlGnBu
num_groups = 6
colors = [cm(x) for x in np.linspace(0.2, 1, num_groups)]
df['name'].unique()

plot_kwargs = {'linewidth': 1}
fig, ax = plt.subplots()
idx = 0

labels = ['50%' ,'40%', '30%', '20%', '10%', 'Clean']
for col in df['name'].unique():
    data = df.loc[df['name'] == col]
    ax.plot(data['idx_T'], data['aucs'], color=colors[idx], label=labels[idx], **plot_kwargs)
    idx += 1


ax.set_xlabel("# of Tokens Observed")
ax.set_ylabel("AUC")
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.legend()

ax.grid(which="both")
plt.show()
plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.95, wspace=0., hspace=0.)
fig.set_size_inches(width, height, forward=True)
wd = "./"
fig.savefig(os.path.join(wd, "visualization/auc-at-t-cp.pdf"))
