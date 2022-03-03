
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def get_dist(a,b):
    return np.sqrt(np.min(np.sum(np.power(a-b,2),axis=1)))

def random_walk():
    p1 = np.load("p1.npy")
    p2 = np.load("p2.npy")
    p3 = np.load("p3.npy")
    for p in [p1,p2,p3]:
        p[:,0:2]=(p[:,0:2]-2)/3
    print("min distance btw 1-2",get_dist(p1,p2))
    print("min distance btw 1-3",get_dist(p1,p3))
    print("min distance btw 2-3",get_dist(p2,p3))
    # p2 = p1.copy()
    # p2[:,0]+=2
    # p3 = p1.copy()
    # p3[:,0]+=4
    # print(p1.shape)
    return  p1.shape[0],[p1]


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps, walks = random_walk()

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
lines = [ax.plot([], [], [])[0] for _ in walks]

# Setting the axes properties
# ax.set(xlim3d=(0, 6), xlabel='X')
# ax.set(ylim3d=(0, 6), ylabel='Y')
# ax.set(zlim3d=(0, 6), zlabel='Z')
ax.set(xlim3d=(-1.3, 1.3), xlabel='X')
ax.set(ylim3d=(-1.3, 1.3), ylabel='Y')
ax.set(zlim3d=(0, 2), zlabel='Z')


# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=1)

plt.show()
# ani.save('decentralize.mp4',  
#           writer = 'ffmpeg', fps = 30)
