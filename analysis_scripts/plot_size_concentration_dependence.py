import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import misc_tools as tools

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 13
LEGENDSIZE = 12
LINEWIDTH = 3
DPI = 300
FIGSIZE = (12, 5) # Figure dimension in inches
bw = 0.25
ALPHA = 0.4
s_nudge = 4
c_nudge = 20
salt_color1 = 'C2'
salt_color2 = 'C0'
salt_color3 = 'C4'

def plot_concentration(axis, data, x, bw, color, label, linewidth, alpha):
    """
    Wrapper function to simplify plotting of salt concentrations.
    """
    kernel = gaussian_kde(data, bw)
    density = kernel(x)
    axis.plot(x, density, lw=linewidth, color=color, label=label)
    axis.fill_between(x, 0.0, density, color=color, alpha=alpha)

def plot_salt_number(axis, data, x, bw, color, label, linewidth, alpha):
    """
    Wrapper function to simplify plotting of salt numbers.
    """
    kernel = gaussian_kde(data, bw)
    density = kernel(x)
    density = density / np.sum(density)
    axis.step(edges, density, where='mid',  lw=linewidth, color=color, label=label)
    axis.fill_between(edges, 0.0, density, color=color, alpha=alpha, step='mid')

#--------Concentration dependence--------#
fig, ax = plt.subplots(1, 2, figsize=FIGSIZE)
file_names = ['out1.nc', 'out2.nc', 'out3.nc']

files = ['../testsystems/waterbox/100mM/60Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color1, '100mM', LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1], salt_conc.flatten(), conc_spread, bw, salt_color1, None, LINEWIDTH, ALPHA)

files = ['../testsystems/waterbox/150mM/60Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color2, '150mM', LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1], salt_conc.flatten(), conc_spread, bw, salt_color2, None, LINEWIDTH, ALPHA)

files = ['../testsystems/waterbox/200mM/60Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color3, '200mM', LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1],salt_conc.flatten(), conc_spread, bw, salt_color3, None, LINEWIDTH, ALPHA)

ax[0].set_yticks([])
ax[0].set_ylabel('Probability', fontsize=FONTSIZE)
ax[0].set_xlabel('Number of salt pairs $N_\mathrm{NaCl}$', fontsize=FONTSIZE)
ax[0].legend(fontsize=FONTSIZE)

ax[1].set_yticks([])
ax[1].set_ylabel('Probability density', fontsize=FONTSIZE)
ax[1].set_xlabel('Concentration $c$ (mM)', fontsize=FONTSIZE)

for i in range(2):
    for label in (ax[i].get_xticklabels()):
        label.set_fontsize(TICKSIZE)
        
fig.suptitle('Macroscopic concentration dependence ({0} waters)'.format(ntotal), fontsize=18)
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.savefig('mac_conc_dependence.png', dpi=DPI)

#--------Box size/water number dependence--------#
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
file_names = ['out1.nc', 'out2.nc', 'out3.nc']

files = ['../testsystems/waterbox/150mM/40Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color1, '40 Å$^3$ ({0} waters)'.format(ntotal), LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1], salt_conc.flatten(), conc_spread, bw/5, salt_color1, None, LINEWIDTH, ALPHA*0.8)

files = ['../testsystems/waterbox/150mM/60Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color2, '60 Å$^3$ ({0} waters)'.format(ntotal), LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1],salt_conc.flatten(), conc_spread, bw, salt_color2, None, LINEWIDTH, ALPHA)

files = ['../testsystems/waterbox/150mM/80Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

edges = np.arange(max(0,np.min(nsalt) - s_nudge), np.max(nsalt) + s_nudge)
plot_salt_number(ax[0], nsalt.flatten(), edges, bw, salt_color3, '80 Å$^3$ ({0} waters)'.format(ntotal), LINEWIDTH, ALPHA)

conc_spread = np.linspace(max(0.0, np.min(salt_conc) - c_nudge), np.max(salt_conc) + c_nudge, 100)
plot_concentration(ax[1], salt_conc.flatten(), conc_spread, bw, salt_color3, None, LINEWIDTH, ALPHA*1.2)

#plt.legend(fontsize=FONTSIZE)
ax[0].set_yticks([])
ax[0].set_ylabel('Probability', fontsize=FONTSIZE)
ax[0].set_xlabel('Number of salt pairs $N_\mathrm{NaCl}$', fontsize=FONTSIZE)
ax[0].legend(fontsize=FONTSIZE)

ax[1].set_yticks([])
ax[1].set_ylabel('Probability density', fontsize=FONTSIZE)
ax[1].set_xlabel('Concentration $c$ (mM)', fontsize=FONTSIZE)

for i in range(2):
    for label in (ax[i].get_xticklabels()):
        label.set_fontsize(TICKSIZE)
        
fig.suptitle('Box size dependence', fontsize=18)
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.savefig('box_size_dependence.png', dpi=DPI)
