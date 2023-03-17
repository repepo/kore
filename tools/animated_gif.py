import imageio
from pygifsicle import optimize

from koreviz import kmode as k
# u = k.kmode('u', 0, nr=500, nphi=500, phase=0)

n = 48
angles = linspace(0,360,n+1)[:-1]

#limits = [u.ur.min(), u.ur.max()]
limits = [-0.0095,0.0095]

filenames = []
for j,angle in enumerate(angles):
    
    u = k.kmode('u', solnum=0, ntheta = 1440, phase=angle*np.pi/180) 
    u.merid(comp='phi',azim=0,colbar=False,limits=limits)
    filename = f'{j}.png'
    savefig(filename,dpi=120)
    filenames.append(filename)
    close('all')
    
# build gif
with imageio.get_writer('imode.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


optimize('imode.gif')
