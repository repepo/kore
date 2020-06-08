'''
Use Ipython for this:

First load the pm_runs dataset:
%run -i ../../tools/getresults.py pm_runs

The run this script:
%run -i figura3.py

'''


import scipy.optimize as sop

def f(x,c,b,m):  
	return c+b*x**m
	



Lec = 10**-2.70
err2c = 0.02
err1c = 0.07

k0 = (err2<err2c)&(err1<err1c)&(bci+bco==2)&(t2p>2)&(Le<Lec)&(ss>=0.7)&(ss<=0.94)&(pm<=-1.8)&(pm>-5)
pm0 = unique(pm[k0])

pm3 = array([])
m3 = array([])
c3 = array([])

for i,pm1 in enumerate(pm0):

	k1 = (err2<err2c)&(err1<err1c)&(bci+bco==2)&(t2p>2)&(Le<Lec)&(ss>=0.7)&(pm==pm1)&(ek<-7)&(ss<=0.94)

	ss1 = unique(ss[k1])
	
	if size(ss1)>3:
	
		x = 10**ss1
		y = zeros(shape(ss1))
	
		for j,ss2 in enumerate(ss1):
	
			k2 =  (err2<err2c)&(err1<err1c)&(bci+bco==2)&(t2p>2)&(Le<Lec)&(ss==ss2)&(pm==pm1)&(ek<-7)&(ss<=0.94)
			if size(scd[k2])>=3:
				y[j] = mean(scd[k2])
			else:
				y[j]=nan
			
		print('pm=',pm1)
		
		idx = isfinite(x)&isfinite(y)
		if sum(idx)>=3:
			m1,b1 = polyfit(log(x[idx]),log(y[idx]),1)
			print(exp(b1),m1)
			m3=r_[m3,m1]
			c3=r_[c3,exp(b1)]
			pm3=r_[pm3,pm1]


# do fits
popt1,pcov1 = sop.curve_fit(f,10**pm3,m3,[0.45,-0.7,0.2])
print('Fit for alpha',popt1)

popt2,pcov2 = sop.curve_fit(f,10**pm3,c3,[3.5,-0.4,0.4])
print('Fit for c',popt2)

# plot results
fig,ax = subplots(2,1,sharex=True,figsize=(6,5))
x = logspace(-5,-1,50)

ax[0].set_title(r'$\sigma\,(P_m,\Lambda,E)=-c\,(P_m)\,\Lambda^{\alpha(P_m)}\,E^{1/2}$',size=12) 
ax[0].plot(10**pm3,c3,'bo',ms=4)
ax[0].plot(x,f(x,popt2[0],popt2[1],popt2[2]),'k--',lw=0.5,label=r'Fit: $c\,(P_m)\simeq 3.396-2.147\,P_m^{0.433}$')
ax[0].set_ylabel(r'$c\,(P_m)$',size=14)
ax[0].set_ylim(3,3.4) 
ax[0].legend(loc=3,frameon=False,fontsize=11)

ax[1].plot(10**pm3,m3,'bo',ms=4)
ax[1].plot(x,f(x,popt1[0],popt1[1],popt1[2]),'k--',lw=0.5,label=r'Fit: $\alpha\,(P_m)\simeq 0.446-0.081\,P_m^{0.44}$')
ax[1].set_ylabel(r'$\alpha\,(P_m)$',size=14)
ax[1].set_xlabel(r'$P_m$',size=14)
ax[1].set_ylim(0.43,0.448)
ax[1].set_xlim(4e-5,2e-2)
ax[1].set_xscale('log')
ax[1].legend(loc=3,frameon=False,fontsize=11)



tight_layout()
show()

'''
ek1 = unique(ek[k0])

m = zeros(shape(ek1))
b = zeros(shape(ek1))


for j,ek2 in enumerate(ek1):

	k =  (err2<0.02)&(err1<0.07)&(bci+bco==2)&(t2p>2)&(Le<Lec)&(ek==ek2)

	x = Lam[k]
	y = scd[k]
	plot(x,y,'.')
	
	m[j],b[j] = polyfit(log(x),log(y),1)
show()

m2 = mean(m)
b2 = mean(b) 
print(mean(exp(b)),mean(m))
'''
