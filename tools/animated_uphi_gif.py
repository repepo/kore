import imageio
from pygifsicle import optimize



n = 48
phases = linspace(0,360,n+1)[:-1]*2*pi/360

limits = [-max(real(uphi)), +max(real(uphi))]


filenames = []
for j,phase in enumerate(phases):
    
    
    uphi0 = uphi*np.exp(-1j*phase)

    plt.figure(1)
    plt.plot(r,np.real(uphi0))
    plt.ylim(limits[0],limits[1])
    plt.show()

    
    filename = f'{j}.png'
    savefig(filename,dpi=120)
    filenames.append(filename)
    
    close(1)
    
# build gif
with imageio.get_writer('imode.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)


optimize('imode.gif')
