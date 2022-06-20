
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (7,5)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#plt.rcParams['font.family'] = 'Times New Roman'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


fig = plt.figure()
alpha = [0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5]
MPJPE = [39.79, 38.63, 38.27, 39.99, 38.94, 38.64, 40.79, 40.72, 41.08, 42.60]
PAMPJPE = [31.48, 30.41, 30.51, 30.96, 30.75, 31.06, 32.14, 32.53, 31.85, 33.66]
plt.plot(alpha, MPJPE, 'ro-', label='Method 1', lw=2)
plt.plot(alpha, PAMPJPE, 'bs-', label='Method 2', lw=2)
plt.legend(loc='best')
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel(r'Evaluation Metric (mm)', fontsize=15)
fig.savefig('test.pdf', format='PDF', dpi=300, bbox_inches='tight')