'''
Use Ipython for this:

Load the results from pm_runs:
%run -i ../../tools/getresults.py pm_runs

And then run this script:
%run -i figuras_2y4.py

'''

rc('text', usetex=True)

n = 6

colormap = cm.plasma
colors = [colormap(i) for i in np.linspace(0, 0.8,n)]



k = (err1<0.05)&(t2p>2)&(ek==-9)&(Lam<40)

pm0 = [-2.0, -2.2, -2.4, -2.6, -2.8, -3.0]


# ------------------ Figure 2

fig,ax = subplots()

for j in range(6):
	
	k1 = k&(pm==pm0[j])
	
	x = Lam[k1]
	y = scd[k1]
	
	k2 = argsort(x)
	
	ax.plot( x[k2], y[k2], c=colors[j], lw=1.7, label=r'${{{}}}$'.format(pm0[j]) )

leg1 = ax.legend(fontsize=10.5,loc='upper left',ncol=1,framealpha=1.0)
leg1.set_title(r'$\log_{10}P_m$',prop={'size':12.5})

x1 = logspace(-1,2,20)	 
line2 = ax.plot(x1,2.7*x1**0.44,'k:',lw=1.3,label=r'$\propto \Lambda^{0.44}$')

ax.hlines(2.62047,1e-3,14,linestyle='--',color='k',lw=1.0)

ax.set_xlabel(r'$\Lambda$',size=16)
ax.set_ylabel(r'$|\sigma|\,E^{-1/2}$',size=16)
ax.set_xscale('log')
ax.set_xlim(1e-3,50)
ax.set_yscale('log')
ax.set_ylim(2,18)

leg2 = ax.legend(handles=line2,fontsize=14,loc='center left',ncol=1,framealpha=1.0, frameon=0)
ax.add_artist(leg1)

ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
ax.tick_params(axis = 'y', which = 'minor', labelleft='off')
ax.set_yticks([2,3,4,5,10])
ax.set_yticklabels(['$2$','$3$','$4$','$5$','$10$'])

ax.text(16,2.53,r'$2.62047$',fontsize=11)

tight_layout()
show()




# ------------------ Figure 4

fig2,ax2 = subplots()

for j in range(6):
	
	k1 = k&(pm==pm0[j])
	
	x = Lam[k1]
	y = Dohm[k1]/Dint[k1]
	
	k2 = argsort(x)
	
	ax2.plot( x[k2], y[k2], c=colors[j], lw=1.7, label=r'${{{}}}$'.format(pm0[j]) )

leg3 = ax2.legend(fontsize=10.5,loc='upper left',ncol=1,framealpha=1.0)
leg3.set_title(r'$\log_{10}P_m$',prop={'size':12.5})

ax2.set_xlabel(r'$\Lambda$',size=16)
ax2.set_ylabel(r'$\mathcal{D}_\eta/\mathcal{D}_\nu$',size=16)
ax2.set_xscale('log')
ax2.set_xlim(1e-3,50)
ax2.set_yscale('log')
ax2.set_ylim(5e-4,1.6)



ax2.tick_params(axis = 'both', which = 'major', labelsize = 12)

tight_layout()
show()


#fig.savefig('figura2.png')
#fig.savefig('figura2.eps')
#fig.savefig('figura2.pdf')

#fig2.savefig('figura4.png')
#fig2.savefig('figura4.eps')
#fig2.savefig('figura4.pdf')
