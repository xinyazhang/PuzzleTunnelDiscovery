import numpy as np
from scipy.interpolate import interp2d
NAN = np.NAN
from scipy.misc import imsave
import texture_format

def _uv_to_pix(u, v, raster_shape):
    '''
    According to glViewPort:
        x_r = (x_ndc+1)*width/2 + x_0
        y_r = (y_ndc+1)*height/2 + y_0
    We also have:
        x_ndc = 2.0 * u - 1.0
        y_ndc = 2.0 * v - 1.0
    Note x_0 = y_0 = 0 for all our configurations.

    Hence:
        x_r = ((2.0 * u - 1.0) + 1) * width / 2 = u * width
    Similarity
        y_r = v * height
    '''
    sizes = np.array(raster_shape)
    # print("halves {} raster_shape {}".format(halves, raster_shape))
    return np.array([u, v]) * sizes

def _bilinear(raster, u, v):
    pix = _uv_to_pix(u, v, raster.shape)
    # print("UV to Pix {} {} -> {}".format(u, v, pix))
    pixi = np.floor(pix).astype(int) # Pix Integer
    r = pix - pixi # Remainder
    def _sample(x, y, no_data=0.0):
        if x < 0 or x >= raster.shape[0]:
            return no_data
        if y < 0 or y >= raster.shape[1]:
            return no_data
        return raster[x, y]
    return _sample(pixi[0], pixi[1]) * (1.0 - r[0]) * (1.0 - r[1]) + \
           _sample(pixi[0] + 1, pixi[1]) * r[0] * (1.0 - r[1]) + \
           _sample(pixi[0], pixi[1]+1) * (1.0 - r[0]) * r[1] + \
           _sample(pixi[0] + 1, pixi[1] + 1) * r[0] * r[1]

class AtlasSampler(object):

    def __init__(self, task_partitioner, geo_type, geo_id, task_id):
        self._geo_type = geo_type
        self._geo_id = geo_id
        tp = self._tp = task_partitioner
        self._atlas2prim = np.load(tp.get_atlas2prim_fn(geo_type))['PRIM']
        d = np.load(tp.get_atlas_fn(geo_type, task_id))
        self._atlas = d[d.keys()[0]] # Load the first element
        print("Atlas resolution {}".format(self._atlas.shape))
        self._nzpix = np.nonzero(self._atlas)
        self._nzpixweight = self._atlas[self._nzpix]
        self._nzpixweight /= np.sum(self._nzpixweight)
        self._nzpix_idx = np.array([i for i in range(len(self._nzpix[0]))], dtype=np.int32)
        self._nzprim, self._nzcount = np.unique(self._atlas2prim[np.nonzero(self._atlas)], return_counts=True)
        # Debugging
        imsave('debug-{}-atlas.png'.format(geo_type), self._atlas)
        binp = np.zeros(shape=self._atlas.shape)
        binp[self._nzpix] = 1.0
        imsave('debug-{}-atlas-bin.png'.format(geo_type), binp)
        imsave('debug-{}-atlas-prim.png'.format(geo_type), self._atlas2prim)
        # print("Atlas nonzero faces {}".format(np.unique(self._atlas2prim[np.nonzero(self._atlas)])))
        X,Y = np.nonzero(self._atlas)
        '''
        # Sanity check
        for x,y in zip(X,Y):
            #assert self._atlas2prim[x,y] >= 0, "_atlas2prim[{},{}] = {}".format(x, y, self._atlas2prim[x,y])
            if self._atlas2prim[x,y] < 0:
                print("{} Atlas {} {} = {}".format(geo_type, x,y, self._atlas2prim[x,y]))
        '''
        # Remove leading -1 elements
        while self._nzprim[0] < 0:
            self._nzprim = np.delete(self._nzprim, 0)
            self._nzcount = np.delete(self._nzcount, 0)
        self._nzcount = self._nzcount.astype(float) / np.sum(self._nzcount)
        print(self._nzprim)

    '''
    sample:
        Importance sampling over the atlas

        Details:
        1. Prefilter non-zero primitives (done in __init__)
        2. Sample within non-zero primitives
        3. Sample over the surface
        4. Check the atlas texture, and reject samples with zero probablity.
    '''
    def sample_old(self, r, unit=True):
        fail = 0
        while True:
            prim = np.random.choice(self._nzprim, p=self._nzcount)
            v3d, normal, uv = r.sample_over_primitive(self._geo_id, prim, return_unit=unit)
            sufuv = texture_format.uv_surface_to_numpy(uv)
            pdf = _bilinear(self._atlas, sufuv[0], sufuv[1])
            '''
            if pdf != 0.0:
                print("pdf {}".format(pdf))
            '''
            if pdf > 0.0:
                break
            fail += 1
        # print("After {} trials, sample at face {} uv {} with pdf {}. Corresponding 3D position {}".format(fail, prim, uv, pdf, v3d))
        return v3d, normal, uv

    def sample(self, r, unit=True):
        fail = 0
        res = self._atlas.shape
        pres = 1.0 / np.array(res) # Pertubation magnitude
        '''
        # SANITY CHECK
        for i in range(1024):
            prim = 13
            v3d, normal, uv = r.sample_over_primitive(self._geo_id, prim, return_unit=unit)
            sufuv = texture_format.uv_surface_to_numpy(uv)
            pdf = _bilinear(self._atlas2prim, sufuv[0], sufuv[1])
            print("[{}] pdf = {} (EXPECT {}) for uv = {}".format(i, pdf, prim, uv))
        return
        '''
        while True:
        #for idx in range(len(self._nzpix[0])):
            idx = np.random.choice(self._nzpix_idx, p=self._nzpixweight)
            pert = pres * np.random.uniform(low=-0.5, high=0.5, size=(2))
            # pert = pres * 0
            pix = np.array([self._nzpix[0][idx], self._nzpix[1][idx]])
            prim = self._atlas2prim[pix[0], pix[1]]
            if prim < 0:
                continue
            uv = pix * pres + pert
            surface_uv = texture_format.uv_numpy_to_surface(uv)
            v3d, normal, valid = r.uv_to_surface(self._geo_id, prim, surface_uv, return_unit=unit)
            # print("uv {} (pix {}) to {} {} {} in prim {}".format(uv, pix, v3d, normal, valid, prim))
            if valid:
                break
        return v3d, normal, uv
