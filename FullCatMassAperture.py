import numpy as np
import healpy as hp
from ShearCatalog import ShearCatalog
from MassAperture import MassApertureMap, polar_angle
from scipy.spatial import cKDTree

class BinCatMassAperture(MassApertureMap):
    def __init__(self, nside=2048):
        super().__init__(nside)
    
    def get_mass_aperture(self, shear_catalog : ShearCatalog, filter, r_theta_cut):
        
        mapE = np.zeros(self._npix)
        mapB = np.zeros(self._npix)
        mask_hp = np.zeros(self._npix)

        radius_rad = r_theta_cut * self._psize_rad
        
        pix_ind = hp.ang2pix(self.nside, shear_catalog.ra, shear_catalog.dec, lonlat = True)
        
        ng = np.bincount(pix_ind, minlength=self._npix)
        shear_catalog.normalise_weights(pix_ind, ng, SUM=False)
        g1 = shear_catalog.gamma1*shear_catalog.weight
        g2 = shear_catalog.gamma2*shear_catalog.weight
        
        pix_vec = self.get_healpix_vec()
        
        mask_hp, ind = self.expand_mask(ng > 0, radius_rad, pix_vec)
        
        vec = hp.ang2vec(shear_catalog.ra, shear_catalog.dec, lonlat = True)
        
        galaxy_KDtree = cKDTree(vec)
        
        npix_nzero = np.size(ind)
        print("npix_zero = ", npix_nzero)
        for i in ind:            
            center = np.radians(hp.pix2ang(self.nside, i, lonlat = True))
            
            neighbors = galaxy_KDtree.query_ball_point(pix_vec[:,i], 2*np.sin(radius_rad/2))
            
            g1_j = g1[neighbors]
            g2_j = g2[neighbors]
            ra_j = shear_catalog.ra[neighbors]
            dec_j = shear_catalog.dec[neighbors]
            
            mapE[i], mapB[i] = self.apply_filter(filter, center, g1_j, g2_j, ra_j, dec_j)
            
        return mapE, mapB, mask_hp
