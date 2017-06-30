import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pymbar import timeseries
import calibration_tools as tools
from glob import glob

# Timeseries parameters
DISCARD = 10
FAST = True

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 12
DPI = 300
FIGSIZE = (5, 5)  # Figure dimension in inches
# The colors for the different water models
t4p_col = 'C0'
t3p_col = 'C1'

#------ Calibration -------
# Loading the calibration and automatically calibrating the chemical potential
files = ['../sams/tip3p/out1.nc', '../sams/tip3p/out2.nc', '../sams/tip3p/out3.nc']
t3p = tools.AutoAnalyzeCalibration(files)

files = ['../sams/tip4pew/out1.nc', '../sams/tip4pew/out2.nc', '../sams/tip4pew/out3.nc']
t4p = tools.AutoAnalyzeCalibration(files)

#---------FIGURE 1-----------
# Relative free energies to add salt
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.15, right=0.95)

x = np.arange(np.max(t3p.nsalt))
ax.errorbar(x, t3p.relative_free_energy, yerr=t3p.error_free_energy*2, label='TIP3P', fmt='o', color=t3p_col)
ax.errorbar(x, t4p.relative_free_energy, yerr=t4p.error_free_energy*2, label='TIP4Pew', fmt='o', color=t4p_col)

ax.plot(x, t3p.relative_free_energy, color=t3p_col)
ax.plot(x, t4p.relative_free_energy, color=t4p_col)

ax.set_xticks(range(0,int(np.max(t3p.nsalt))+1,2))
ax.set_xlabel('Number of salt present', fontsize=FONTSIZE)
ax.set_ylabel('Relative free energy (kT)', fontsize=FONTSIZE)
ax.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('relative_free_energies.png', dpi=300)


# FIGURE 2: Titration curves
#------ Extracting the data used to validate the calibration data ------#

# Loading and analysing all the tip3p data
t3p_files = glob('../equilibrium_staging/tip3p/*/out.nc')
t3p_concentration, t3p_standard_error, t3p_delta_mu = tools.read_concentration(t3p_files, discard=DISCARD, fast=FAST)

# Only loading the tip4pew data that lies within the tip3p range for pretty plotting
folders = ['deltamu_315.78/', 'deltamu_316.91/', 'deltamu_317.93/']
t4p_files = ['../equilibrium_staging/tip4pew/' + f + 'out.nc' for f in folders]
t4p_concentration, t4p_standard_error, t4p_delta_mu = tools.read_concentration(t4p_files, discard=DISCARD, fast=FAST)

# Generating confidence intervals for the relationship between delta mu and concentration.
plot_mus = np.linspace(np.min(t3p_delta_mu), np.max(t3p_delta_mu))

t3p_pred_concentration, t3p_pred_spread = t3p.predict_ensemble_concentrations(deltachems=plot_mus, nsamples=500)
t3p_lower = np.percentile(t3p_pred_spread, q=2.5, axis=1)
t3p_upper = np.percentile(t3p_pred_spread, q=97.5, axis=1)

t4p_pred_concentration, t4p_pred_spread = t4p.predict_ensemble_concentrations(deltachems=plot_mus, nsamples=500)
t4p_lower = np.percentile(t4p_pred_spread, q=2.5, axis=1)
t4p_upper = np.percentile(t4p_pred_spread, q=97.5, axis=1)

# Generating the figure
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111)
fig.subplots_adjust(right=0.95)

# Plotting the calibrated salt concentration with confidence interval
ax.plot(plot_mus, t3p_pred_concentration,color=t3p_col, label='TIP3P calibration', lw=2)
ax.fill_between(x=plot_mus, y1=t3p_lower, y2=t3p_upper, color=t3p_col, alpha=0.3, lw=0)

ax.plot(plot_mus, t4p_pred_concentration,color=t4p_col, label='TIP4Pew calibration', lw=2)
ax.fill_between(x=plot_mus, y1=t4p_lower, y2=t4p_upper, color=t4p_col, alpha=0.3, lw=0)

# Plotting the observed concentration with estimated 95% confidence interval
ax.errorbar(t3p_delta_mu, t3p_concentration,yerr=2*t3p_standard_error, fmt='o', color='black', label='TIP3P observed', lw=2.5, zorder=3)
ax.errorbar(t4p_delta_mu, t4p_concentration,yerr=2*t4p_standard_error, fmt='o', color='grey', label='TIP4Pew observed', lw=2.5, zorder=3)

ax.set_xlabel('Chemical potential (kT)', fontsize=FONTSIZE)
ax.set_ylabel('Concentration (M)', fontsize=FONTSIZE)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

ax.legend(fontsize=LEGENDSIZE)
plt.savefig('salt-water_titration.png', dpi=300)

