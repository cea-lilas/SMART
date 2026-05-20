import numpy as np
import healpy as hp
from ShearCatalog import ShearCatalog
from MassAperture import MassApertureMap, polar_angle

class PixelMassAperture(MassApertureMap):
    def __init__(self, nside=2048):
        super().__init__(nside)
        self._barycenters = False
        self._squares = False
        self._done_pixelisation = False
    
    def pixelise_catalog(self, shear_catalog : ShearCatalog, SUM = False, return_squares=True, return_barycenters=False):
        """
        Groups galaxy shear data into pixels.

        Args:
            shear_catalog (ShearCatalog): A ShearCatalog object containing the galaxy shear data to be pixelised.
            self.nside (int): healpix self.nside parameter  
            SUM (Boolean, optional): If True, group the pixels by summing galaxy shears, averages them otherwise. Defaults to False.
        """

        pix_ind = hp.ang2pix(self.nside, shear_catalog.ra, shear_catalog.dec, lonlat=True, nest=False)
        ng = np.bincount(pix_ind, minlength=self._npix)
        
        shear_catalog.normalise_weights(pix_ind, ng, SUM=SUM)
        
        self.gamma1 = np.zeros(self._npix, dtype=np.float64)
        self.gamma2 = np.zeros(self._npix, dtype=np.float64)
        
        np.add.at(self.gamma1, pix_ind, shear_catalog.gamma1*shear_catalog.weight)
        np.add.at(self.gamma2, pix_ind, shear_catalog.gamma2*shear_catalog.weight)

        if return_squares:
            self._squares = True
            self.gamma1_sq = np.zeros(self._npix, dtype=np.float64)
            self.gamma2_sq = np.zeros(self._npix, dtype=np.float64)
            np.add.at(self.gamma1_sq, pix_ind, (shear_catalog.gamma1*shear_catalog.weight)**2)
            np.add.at(self.gamma2_sq, pix_ind, (shear_catalog.gamma2*shear_catalog.weight)**2)
        
        self.mask = ng > 0

        if return_barycenters:
            self._barycenters = True
            self.ra_c = np.zeros(self._npix, dtype=np.float64)
            self.dec_c = np.zeros(self._npix, dtype=np.float64)
            np.add.at(self.ra_c, pix_ind, shear_catalog.ra)
            self.ra_c[self.mask] = np.radians(self.ra_c[self.mask]/ng[self.mask])
            np.add.at(self.dec_c, pix_ind, shear_catalog.dec)
            self.dec_c[self.mask] = np.radians(self.dec_c[self.mask]/ng[self.mask])
            
        self.ra, self.dec = self.get_healpix_ra_dec()
        self._done_pixelisation = True
    
    def check_pixelisation(self, shear_catalog, SUM, return_squares, return_barycenters):
        if not self._done_pixelisation:
            if shear_catalog is None:
                raise ValueError("No shear catalog provided for pixelisation.")
            self.pixelise_catalog(shear_catalog, SUM, return_squares, return_barycenters)

    def pixelised_shear(self, shear_catalog=None, SUM = False, return_squares=True, return_barycenters=False):
        self.check_pixelisation(shear_catalog, SUM, return_squares, return_barycenters)
        return self.ra, self.dec, self.gamma1, self.gamma2
    
    def get_mass_aperture(self, filter, r_theta_cut, shear_catalog=None, SUM = False, return_squares=True, return_barycenters=False):
        self.check_pixelisation(shear_catalog, SUM, return_squares, return_barycenters)
        
        if filter is None:
            filter = self.get_jarvis(r_theta_cut)

        radius_rad = r_theta_cut * self._psize_rad
        
        mapE = np.zeros(self._npix)
        mapB = np.zeros(self._npix)
        map_vnoise = np.zeros(self._npix)
            
        vecs = self.get_healpix_vec()
            
        mask_hp, ind = self.expand_mask(self.mask, radius_rad, vecs)
        
        for i in ind:
            # Select the healpix pixels iself.nside the aperture of radius_rad
            neighbors = hp.query_disc(self.nside, vecs[:, i], radius_rad)
            neighbors = neighbors[neighbors != i]
            center = (self.ra[i], self.dec[i])
            
            ra_j, dec_j, gamma1_j, gamma2_j, gamma1_sq_j, gamma2_sq_j = self.query_neighbors(neighbors)

            mapE[i], mapB[i], map_vnoise[i] = self.apply_filter(filter, center, gamma1_j, gamma2_j, ra_j, dec_j, gamma1_sq_j, gamma2_sq_j)

        return mapE, mapB, map_vnoise, mask_hp