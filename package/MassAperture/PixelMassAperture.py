import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from .MassAperture import MassApertureMap, polar_angle
from time import time

class PixelMassAperture(MassApertureMap):
    def __init__(self, nside=2048, verbosity=1):
        super().__init__(nside, verbosity)
        self._barycenters = False
        self._done_pixelisation = False
    
    def pixelise_catalog(self, shear_catalog : ShearCatalog, SUM = False, return_squares=True, return_barycenters=False):
        """
        Groups galaxy shear data into pixels.

        Args:
            shear_catalog (ShearCatalog): A ShearCatalog object containing the galaxy shear data to be pixelised.
            self.nside (int): healpix self.nside parameter  
            SUM (Boolean, optional): If True, group the pixels by summing galaxy shears, averages them otherwise. Defaults to False.
        """
        self.verbose_print(1, f"Pixelising catalog : npix = {self._npix}")
        start_time = time()
        pix_ind = hp.ang2pix(self.nside, shear_catalog.ra, shear_catalog.dec, lonlat=True)
        ng = np.bincount(pix_ind, minlength=self._npix)
        
        shear_catalog.normalise_weights(self._npix, pix_ind, ng, SUM=SUM)
        
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
        self.verbose_print(2, f"Pixelisation done in {time() - start_time:.2f} seconds.")

    def check_pixelisation(self, shear_catalog, SUM, return_squares, return_barycenters):
        if not self._done_pixelisation:
            if shear_catalog is None:
                raise ValueError("No shear catalog provided for pixelisation.")
            self.pixelise_catalog(shear_catalog, SUM, return_squares, return_barycenters)

    def pixelised_shear(self, shear_catalog=None, SUM = False, return_squares=True, return_barycenters=False):
        self.check_pixelisation(shear_catalog, SUM, return_squares, return_barycenters)
        return self.ra, self.dec, self.gamma1, self.gamma2

    def query_neighbors(self, i, vecs, radius_rad):
        # Select the healpix pixels iself.nside the aperture of radius_rad
        neighbors = hp.query_disc(self.nside, vecs[:, i], radius_rad)
        neighbors = neighbors[neighbors != i]
        
        center = (self.ra[i], self.dec[i])
        
        gamma1_j = self.gamma1[neighbors]
        gamma2_j = self.gamma2[neighbors]
        
        if self._barycenters:
            ra_j = self.ra_c[neighbors]
            dec_j = self.dec_c[neighbors]
        else:
            ra_j = self.ra[neighbors]
            dec_j = self.dec[neighbors]

        if self._squares:
            gamma1_sq_j = self.gamma1_sq[neighbors]
            gamma2_sq_j = self.gamma2_sq[neighbors]
        else:
            gamma1_sq_j = None
            gamma2_sq_j = None
        
        return center, ra_j, dec_j, gamma1_j, gamma2_j, gamma1_sq_j, gamma2_sq_j

    def initialise_mass_aperture(self, shear_catalog=None, SUM = False, return_squares=True, return_barycenters=False):
        self.check_pixelisation(shear_catalog, SUM, return_squares, return_barycenters)
        return {}