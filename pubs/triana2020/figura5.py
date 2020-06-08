'''
Usage:

In Ipython do:
%run -i figura5.py

'''
import scipy.io as sio


rc('text', usetex=True)

n = 5

colormap = cm.plasma
colors = [colormap(i) for i in np.linspace(0, 0.9,n)]

fig, ax = subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(8,6))

ek = array([-8.0,-8.5,-9.0,-9.5,-10.0])





	
ax[0,0].set_yscale('log')	
ax[0,0].set_ylabel(r'$\mathcal{D}_\nu(r)/\mathcal{D}_\nu(r=1)$',size=14)
#ax[0,0].set_title(r'$\Lambda=10^{-3}$',x=0.22,y=0.84,fontsize=10)
ax[0,0].text(2,0.2,r'Viscous dissipation, $\Lambda=10^{-3}$',fontsize=11)

d = sio.loadmat('Dint_Pm-3_Lambda-3.mat')
r = d['r']
dint = d['Dint']	
for j in range(size(ek)):		
	ax[0,0].plot( (1-r[:,j])/sqrt(10**ek[j]), dint[:,j]/max(dint[:,j]), '-', c=colors[j], lw=1.2, label=r'${{{}}}$'.format(ek[j]))
		

ax[0,1].set_yscale('log')
#ax[0,1].set_title(r'$\Lambda=10^{0.7}\sim 5$',x=0.22,y=0.84,fontsize=11)
ax[0,1].text(1.1,0.2,r'Viscous dissipation, $\Lambda=10^{0.7}$',fontsize=11)

d = sio.loadmat('Dint_Pm-3_Lambda0.7.mat')
r = d['r']
dint = d['Dint']
for j in range(size(ek)):		
	ax[0,1].plot( (1-r[:,j])/sqrt(10**ek[j]), dint[:,j]/max(dint[:,j]), '-', c=colors[j], lw=1.2, label=r'${{{}}}$'.format(ek[j]))


ax[1,0].set_yscale('log')
ax[1,0].set_ylabel(r'$\mathcal{D}_\eta(r)/\mathcal{D}_\eta(r=1)$',size=14)
#ax[1,0].set_title(r'$\Lambda=10^{-3}$',x=0.22,y=0.84,fontsize=11)
ax[1,0].text(1.2,0.4,r'Ohmic dissipation, $\Lambda=10^{-3}$',fontsize=11)
d = sio.loadmat('Dohm_Pm-3_Lambda-3.mat')
r = d['r']
dint = d['Dohm']
for j in range(size(ek)):		
	ax[1,0].plot( (1-r[:,j])/sqrt(10**ek[j]), dint[:,j]/max(dint[:,j]), '-', c=colors[j], lw=1.2, label=r'${{{}}}$'.format(ek[j]))

ax[1,1].set_yscale('log')
#ax[1,1].set_title(r'$\Lambda=10^{0.7}\sim 5$',x=0.22,y=0.84,fontsize=11)
ax[1,1].text(1.1,0.4,r'Ohmic dissipation, $\Lambda=10^{0.7}$',fontsize=11)
d = sio.loadmat('Dohm_Pm-3_Lambda0.7.mat')
r = d['r']
dint = d['Dohm']
for j in range(size(ek)):		
	ax[1,1].plot( (1-r[:,j])/sqrt(10**ek[j]), dint[:,j]/max(dint[:,j]), '-', c=colors[j], lw=1.2, label=r'${{{}}}$'.format(ek[j]))







ax[0,0].set_xlim(0,15)
ax[0,1].set_xlim(0,15)
ax[0,0].set_ylim(1e-7,1)	
ax[1,0].set_ylim(5e-5,1)
		
ax[1,0].set_xlabel(r'$(1-r)E^{-1/2}$',size=13)
ax[1,1].set_xlabel(r'$(1-r)E^{-1/2}$',size=13)

ax[0,1].legend(title=r'$\log_{10}E$',fontsize=10,loc='upper right',ncol=1,framealpha=1.0)
tight_layout()
show()

