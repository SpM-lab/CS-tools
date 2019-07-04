import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib

x = np.arange(0.01, 10, 0.05)
y = np.arange(0.01, 10, 0.05)
X, Y = np.meshgrid(x, y)
m = np.array([1.0, 2.0])

Z = X + Y - m[0] - m[1] - X * np.log(X/m[0]) - Y * np.log(Y/m[1])

fig = plt.figure()
ax = plt.axes()
plt.pcolormesh(X, Y, Z, cmap='magma')
pp=plt.colorbar (orientation="vertical")
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

cont=plt.contour(X,Y,Z,8,vmin=-1,vmax=1, colors=['black'])
cont.clabel(fmt='%1.1f', fontsize=16)
plt.plot(1.0,2.0,marker='.',markersize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.tick_params(which='major', labelsize=16)
plt.gca().set_aspect('equal')

ppdf = PdfPages('information_entropy.pdf')
ppdf.savefig(fig,bbox_inches="tight", pad_inches=0.0)
ppdf.close()
