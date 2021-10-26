import imageio
from pygifsicle import optimize

# from koreviz import kmode as k
# u = k.kmode('u', 0, nr=500, nphi=500)

n = 48
angles = linspace(0,180,n+1)[:-1]

limits = [u.ur.min(), u.ur.max()]


filenames = []
for j,angle in enumerate(angles):
    
    u.merid(comp='r',azim=angle,colbar=False,limits=limits)
    filename = f'{j}.png'
    savefig(filename,dpi=120)
    filenames.append(filename)
    
# build gif
with imageio.get_writer('imode.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


optimize('imode.gif')
