import numpy as np
import sys
import random

from scipy.interpolate import interp2d
NAN = np.NAN
from imageio import imwrite as imsave
import texture_format

from . import matio

def _save_green(ofn, ns):
    gatex = np.zeros(shape=(ns.shape[0], ns.shape[1], 3))
    gatex[:,:,1] = ns
    imsave(ofn, gatex)

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

class SingleChannelAtlasSampler(object):

    def __init__(self, atlas2prim, atlas, geo_type, geo_id):
        self._geo_type = geo_type
        self._geo_id = geo_id
        # FIXME: Notify users to run corresponding commands when files not found
        self._atlas2prim = atlas2prim
        self._atlas = atlas
        #np.clip(self._atlas, 0.0, None, out=self._atlas) # Clips out negative weights
        print("RAW ATLAS Sum {} Max {} Min {} Mean {} Stddev {}".format(
            np.sum(self._atlas),
            np.max(self._atlas),
            np.min(self._atlas),
            np.mean(self._atlas),
            np.std(self._atlas)
            ))
        #self._atlas -= np.min(self._atlas) # All elements must be non-negative
        np.clip(self._atlas, a_min=0.0, a_max=None, out=self._atlas) # All elements must be non-negative
        # Match the visualization results
        self._atlas /= np.max(self._atlas)
        self._atlas *= 255.0
        self._atlas = self._atlas.astype(np.uint).astype(np.float32)
        self._atlas[self._atlas < 32.0] = 0.0
        print("Atlas resolution {}".format(self._atlas.shape))
        self._nzpix = np.nonzero(self._atlas)
        self._nzpixweight = self._atlas[self._nzpix]
        nzsum = np.sum(self._nzpixweight)
        print("NZ pix num {} {} sum {}".format(self._nzpix[0].shape, self._nzpix[1].shape, nzsum))
        print("NZ pix coord maxs {} {}".format(np.max(self._nzpix[0]), np.max(self._nzpix[1])))
        print("NZ pix coord mins {} {}".format(np.min(self._nzpix[0]), np.min(self._nzpix[1])))
        self._nzpixweight /= nzsum
        self._nzpix_idx = np.array([i for i in range(len(self._nzpix[0]))], dtype=np.int32)
        print("ATLAS Sum {} Max {} Min {} Mean {} Stddev {}".format(
            np.sum(self._atlas),
            np.max(self._atlas),
            np.min(self._atlas),
            np.mean(self._atlas),
            np.std(self._atlas)
            ))
        print("NZPIXWEIGHT Sum {} Max {} Min {} Mean {} Stddev {}".format(
            np.sum(self._nzpixweight),
            np.max(self._nzpixweight),
            np.min(self._nzpixweight),
            np.mean(self._nzpixweight),
            np.std(self._nzpixweight)
            ))
        # print("nz {}".format(self._nzpix))
        # print("atlas2prim[nz] {}".format(self._atlas2prim[self._nzpix]))
        self._nzprim, self._nzcount = np.unique(self._atlas2prim[self._nzpix], return_counts=True)
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

        self._debug_nsample = None
        self._debug_vsample = None
        self._debug_v3d = []
        self._debug_allnsample = None

    def enable_debugging(self):
        self._debug_nsample = np.zeros(shape=self._atlas.shape, dtype=np.int32)
        self._debug_vsample = np.zeros(shape=self._atlas.shape, dtype=np.int32)
        self._debug_v3d = []
        self._debug_allnsample = np.zeros(shape=self._atlas.shape, dtype=np.int32)

    def dump_debugging(self, prefix='./'):
        #np.savez('{}debug-{}-nsample.npz'.format(prefix, self._geo_type), ATEX=self._debug_nsample)
        _save_green('{}debug-{}-nsample.png'.format(prefix, self._geo_type), self._debug_nsample)
        _save_green('{}debug-{}-vsample.png'.format(prefix, self._geo_type), self._debug_vsample)
        matio.savetxt('{}debug-{}-v3d.txt'.format(prefix, self._geo_type), self._debug_v3d)

    def debug_surface_sampler(self, ofn):
        fatlas = np.copy(self._atlas)
        for i in range(12):
            print("FATLAS Sum {} Max {} Min {} Mean {} Median {} Stddev {}".format(
                np.sum(fatlas),
                np.max(fatlas),
                np.min(fatlas),
                np.mean(fatlas),
                np.median(fatlas),
                np.std(fatlas)
                ))
            nzmedian = np.median(fatlas[fatlas.nonzero()])
            fatlas[fatlas < nzmedian] = 0.0
            print("{} fatlas nz {}".format(i, len(np.nonzero(fatlas)[0])))
        ns = np.zeros(shape=self._atlas.shape, dtype=np.int32)
        ns[fatlas.nonzero()] = 255
        _save_green(ofn, ns)
        return
        li_indices = np.random.choice(self._nzpix_idx, size=(262144), p=self._nzpixweight)
        # li_indices = self._nzpixweight.argsort()[:64]
        bbins = np.bincount(li_indices)
        print('bbins.shape {}'.format(bbins.shape))
        pi_indices = np.array([self._nzpix[0][:len(bbins)], self._nzpix[1][:len(bbins)]])
        print('pi_indices.shape {}'.format(pi_indices.shape))
        ns = np.zeros(shape=self._atlas.shape, dtype=np.int32)
        ns[pi_indices[0], pi_indices[1]] += bbins
        _save_green(ofn, ns)

    def get_top_k_surface_tups(self, r, top_k):
        li_indices = self._nzpixweight.argsort()
        res = self._atlas.shape
        pres = 1.0 / np.array(res) # Pertubation magnitude
        tups = []
        for idx in li_indices:
            pix = np.array([self._nzpix[0][idx], self._nzpix[1][idx]])
            prim = self._atlas2prim[pix[0], pix[1]]
            if prim < 0:
                print("Cannot find prim from pix {}".format(pix))
                continue
            uv = pix * pres
            print("probing {}".format(idx))
            sys.stdout.flush()
            surface_uv = texture_format.uv_numpy_to_surface(uv)
            v3d, normal, valid = tup = r.uv_to_surface(self._geo_id, prim, surface_uv, return_unit=True)
            if valid:
                tups.append(tup)
                self._debug_v3d.append(v3d)
                if len(tups) >= top_k:
                    break
        return tups

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
        # print('NZPIX {}'.format(self._nzpixweight))
        while True:
        #for idx in range(len(self._nzpix[0])):
            idx = np.random.choice(self._nzpix_idx, p=self._nzpixweight)
            pert = pres * np.random.uniform(low=-0.5, high=0.5, size=(2))
            # pert = pres * 0
            pix = np.array([self._nzpix[0][idx], self._nzpix[1][idx]])
            prim = self._atlas2prim[pix[0], pix[1]]
            if prim < 0:
                continue
            if self._debug_nsample is not None:
                self._debug_nsample[pix[0], pix[1]] += 1
            uv = pix * pres + pert
            surface_uv = texture_format.uv_numpy_to_surface(uv)
            v3d, normal, valid = r.uv_to_surface(self._geo_id, prim, surface_uv, return_unit=unit)
            # print("uv {} (pix {}) to {} {} {} in prim {}".format(uv, pix, v3d, normal, valid, prim))
            if valid:
                break
        if self._debug_vsample is not None:
            self._debug_vsample[pix[0], pix[1]] += 1
        if self._debug_v3d is not None:
            self._debug_v3d.append(v3d)
        return v3d, normal, uv, prim

class AtlasSampler(object):
    def __init__(self, atlas2prim_fn, surface_prediction_fn, geo_type, geo_id):
        self._atlas2prim = np.load(atlas2prim_fn)['PRIM']
        if surface_prediction_fn is None:
            self._mc_atlas = np.clip(self._atlas2prim + 1, a_min=0, a_max=1).astype(np.float64)
        else:
            self._mc_atlas = matio.load(surface_prediction_fn, key='ATEX')
        if len(self._mc_atlas.shape) == 2:
            self._mc_atlas = np.expand_dims(self._mc_atlas, 2)
        self._as = [SingleChannelAtlasSampler(self._atlas2prim, self._mc_atlas[:,:,i], geo_type, geo_id) for i in range(self._mc_atlas.shape[2]) ]

    def sample(self, r, unit=True):
        ats = random.choice(self._as)
        return ats.sample(r, unit)
