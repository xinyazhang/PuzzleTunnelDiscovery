import pyosr
import numpy as np
import matplotlib.pyplot as plt

pyosr.init()
pyosr.create_gl_context(pyosr.create_display())

r=pyosr.Renderer()
r.setup()
r.loadModelFromFile('../res/alpha/robot.obj')
r.angleModel(30.0, 30.0)
pix = r.render_depth_to_buffer()
r.teardown()
pyosr.shutdown()

w = r.pbufferWidth
h = r.pbufferHeight

img = np.array(pix, dtype=np.float32)
img = img.reshape(w, h)

plt.pcolor(img)
plt.show()
