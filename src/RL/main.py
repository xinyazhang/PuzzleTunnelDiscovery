import pyosr
import numpy as np
import matplotlib.pyplot as plt

pyosr.init()
pyosr.create_gl_context(pyosr.create_display())

r=pyosr.Renderer()
r.setup()
r.loadModelFromFile('../res/alpha/env-1.2.obj')
r.loadRobotFromFile('../res/alpha/robot.obj')
r.scaleToUnit()
r.angleModel(0.0, 0.0)
r.default_depth = 0.0
r.views = np.array([[30.0, float(i)] for i in range(0, 360, 30)], dtype=np.float32)
print(r.views)
print(r.views.shape)
r.state = np.array([0.2, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
mvpix = r.render_mvdepth_to_buffer()
w = r.pbufferWidth
h = r.pbufferHeight
img = mvpix.reshape(w * r.views.shape[0], h)
print(img.shape)
plt.pcolor(img)
plt.show()

r.teardown()
pyosr.shutdown()
