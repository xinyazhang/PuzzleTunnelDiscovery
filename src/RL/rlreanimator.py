import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ReAnimator(object):

    def __init__(self):
        self.im = None

    def perform(self, rgb):
        if self.im is None:
            print('rgb {}'.format(rgb.shape))
            self.im = plt.imshow(rgb)
        else:
            self.im.set_array(rgb)

def reanimate(generator, fps=5):
    ra = ReAnimator()
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, ra.perform, interval=1000/fps, frames=generator, repeat=False)
    plt.show()
