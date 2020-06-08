'''
Use Ipython for this:

Load the results from pm_runs:
%run -i ../../tools/getresults.py s_runs

And then run this script:
%run -i figura1.py

'''


rc('text', usetex=True)

n = 10

colormap = cm.plasma
colors = [colormap(i) for i in np.linspace(0, 0.95,n)]

fig, ax = subplots(nrows=3,ncols=1,sharex=True,figsize=(6,8))

pm0 = array([-3.5,-4.5,-5.5])

for i in range(3):

	k1 = (err1<0.07)&(t2p>2)&(pm==pm0[i])&(ss<0.85)

	x1 = Ek[k1]
	y1 = scd[k1]
	z1 = ss[k1]
	
	z0 = unique(z1)

	ax[i].set_xscale('log')
	ax[i].set_yscale('log')
	ax[i].set_xlim(10**-11,10**-8.5)
	ax[i].set_ylim(3,30)
	ax[i].set_ylabel(r'$\sigma\,E^{-1/2}$',size=14)
	ax[i].tick_params(axis = 'y', which = 'minor', labelleft='off')
	ax[i].set_yticks([3,4,5,10,20,30])
	ax[i].set_yticklabels(['$3$','$4$','$5$','$10$','$20$','$30$'])
	
	#ax[i].title.set_text(r'$P_m=10^{{{}}}$'.format(pm0[i]))
	ax[i].set_title(r'$P_m=10^{{{}}}$'.format(pm0[i]),x=0.11,y=0.84,fontsize=12.5)
	ax[i].tick_params(axis = 'both', which = 'major', labelsize = 11)
	
	
	for j in range(size(z0)):
		
		k2 = z1==z0[j]
	
		x2 = x1[k2]
		y2 = y1[k2]
		z2 = z1[k2]
		
		k3 = argsort(x2)		
		
		#ax[i].plot(x2[k3],y2[k3],'.-',c=colors[j],label=r'$10^{{{}}}$'.format(z0[j]))
		ax[i].plot(x2[k3],y2[k3],'.-',c=colors[j],lw=1.7,ms=4,label=r'${{{}}}$'.format(z0[j]))

ax[2].set_xlabel(r'$E$',size=14)
ax[0].legend(title=r'$\log_{10}\Lambda$',fontsize=9,loc='upper right',ncol=2,framealpha=1.0)
tight_layout()
show()

