import numpy as np
import healpy as hp
from ShearCatalog import ShearCatalog
from MassAperture import MassApertureMap, polar_angle

class BinCatMassAperture(MassApertureMap):
    def __init__(self, nside=2048):
        super().__init__(nside)
        
    def bin_catalog(self, shear_catalog : ShearCatalog, SUM = False):
        pix_ind = hp.ang2pix(self.nside, ra, dec, nest=False, lonlat=True)
        ngal_pix = np.bincount(pix_ind, minlength=self._npix)
        shear_catalog.normalise_weights(pix_ind, ngal_pix, SUM=SUM)
        
        ra, dec = np.radians(shear_catalog.ra), np.radians(shear_catalog.dec)
            
        pixel_order = np.argsort(pix_ind)
        offsets = np.r_[0, np.cumsum(ngal_pix), self._ngal]
        
        all_gals = np.empty((self._ngal, 4), dtype=np.float64)

        all_gals[:, 0] = ra[pixel_order]
        all_gals[:, 1] = dec[pixel_order]
        
        mask = (ngal_pix > 0)
            
        all_gals[:, 2] = shear_catalog.gamma1[pixel_order] * shear_catalog.weight[pixel_order]
        all_gals[:, 3] = shear_catalog.gamma2[pixel_order] * shear_catalog.weight[pixel_order]
        
        return all_gals, offsets, mask
    
    def get_mass_aperture(self, shear_catalog, filter, r_theta_cut):
        ra, dec = self.get_healpix_ra_dec()
        
        all_gals, offsets, mask = self.bin_catalog(shear_catalog)
        
        if filter is None:
            filter = self.get_jarvis(r_theta_cut)
        radius_rad = r_theta_cut * self._psize_rad
        mapE = np.zeros(self._npix)
        mapB = np.zeros(self._npix)
        map_vnoise = np.zeros(self._npix)
        
        vecs = self.get_healpix_vec()
        
        mask_hp, ind = self.expand_mask(mask, radius_rad, vecs)
        
        for i in ind:
            neighbors = hp.query_disc(self.nside, vecs[:, i], radius_rad)
            
            neighbor_lenghts = offsets[neighbors+1] - offsets[neighbors]
            neighbor_data = np.empty((sum(neighbor_lenghts), 4), dtype=all_gals.dtype)
            
            pos = 0
            for n in range(len(neighbors)):
                neighbor_data[pos: pos + neighbor_lenghts[n], :] = all_gals[offsets[neighbors[n]]: offsets[neighbors[n]+1], :]
                pos += neighbor_lenghts[n]
            
            ra_j = neighbor_data[:, 0]
            dec_j = neighbor_data[:, 1]
            gamma1_j = neighbor_data[:, 2]
            gamma2_j = neighbor_data[:, 3]
            gamma1_j_sq = gamma1_j**2
            gamma2_j_sq = gamma2_j**2
            
            center = (ra[i], dec[i])
            
            mapE[i], mapB[i], map_vnoise[i] = self.apply_filter(filter, center, gamma1_j, gamma2_j, ra_j, dec_j, gamma1_sq=gamma1_j_sq, gamma2_sq=gamma2_j_sq)
            
        return mapE, mapB, map_vnoise, mask_hp