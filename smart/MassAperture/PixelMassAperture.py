import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from .MassAperture import MassApertureMap
from time import time


class PixelMassAperture(MassApertureMap):
    def __init__(self, nside=2048, verbosity=1):
        super().__init__(nside, verbosity)
        self._barycenters = False
        self._done_pixelisation = False

    def pixelise_catalog(
            self,
            shear_catalog: ShearCatalog,
            SUM=False,
            return_squares=True,
            return_barycenters=False):
        """
        Groups galaxy shear data into pixels.

        Args:
            shear_catalog : ShearCatalog
                A ShearCatalog object containing the galaxy shear data to be pixelised.
            SUM : Boolean, optional
                If True, group the pixels by summing galaxy shears and normalising by the overall average, averages them per pixel otherwise. Defaults to False.
        """
        self._sum = SUM
        self.verbose_print(1, f"Pixelising catalog : npix = {self._npix}")
        start_time = time()
        pix_ind = hp.ang2pix(
            self.nside,
            shear_catalog.ra,
            shear_catalog.dec,
            lonlat=True)
        ng = np.bincount(pix_ind, minlength=self._npix)

        self.ng_weight = self.get_healpix_weights(shear_catalog, pix_ind)

        if not SUM:
            shear_catalog.normalise_weights(pix_ind, self.ng_weight)

        self.gamma1 = np.zeros(self._npix, dtype=np.float64)
        self.gamma2 = np.zeros(self._npix, dtype=np.float64)

        np.add.at(
            self.gamma1,
            pix_ind,
            shear_catalog.gamma1 *
            shear_catalog.weight)
        np.add.at(
            self.gamma2,
            pix_ind,
            shear_catalog.gamma2 *
            shear_catalog.weight)

        if return_squares:
            self._squares = True
            self.gamma1_sq = np.zeros(self._npix, dtype=np.float64)
            self.gamma2_sq = np.zeros(self._npix, dtype=np.float64)
            np.add.at(
                self.gamma1_sq,
                pix_ind,
                (shear_catalog.gamma1 * shear_catalog.weight)**2)
            np.add.at(
                self.gamma2_sq,
                pix_ind,
                (shear_catalog.gamma2 * shear_catalog.weight)**2)

        self.mask = ng > 0

        if return_barycenters:
            self._barycenters = True
            self.ra_c = np.zeros(self._npix, dtype=np.float64)
            self.dec_c = np.zeros(self._npix, dtype=np.float64)
            np.add.at(self.ra_c, pix_ind, shear_catalog.ra)
            self.ra_c[self.mask] = np.radians(
                self.ra_c[self.mask] / ng[self.mask])
            np.add.at(self.dec_c, pix_ind, shear_catalog.dec)
            self.dec_c[self.mask] = np.radians(
                self.dec_c[self.mask] / ng[self.mask])

        self.ra, self.dec = self.get_healpix_ra_dec()
        self._done_pixelisation = True
        self.verbose_print(
            2, f"Pixelisation done in {self.time_to_string(time() - start_time)}")

    def check_pixelisation(
            self,
            shear_catalog=None,
            SUM=False,
            return_squares=True,
            return_barycenters=False):
        """
        Checks if the catalog has been pixelised, and pixelises it if not, if shear catalog is provided.

        Args:
            shear_catalog : ShearCatalog, optional
                A ShearCatalog object containing the galaxy shear data to be pixelised if not already done. If None and pixelisation has not been done, raises an error. Defaults to None.
            SUM, return_squares, return_barycenters : Boolean, optional
                Parameters to pass to the pixelisation function if pixelisation needs to be done. Defaults to False, True, False respectively.
        """
        if not self._done_pixelisation:
            if shear_catalog is None:
                raise ValueError("No shear catalog provided for pixelisation.")
            self.pixelise_catalog(
                shear_catalog,
                SUM,
                return_squares,
                return_barycenters)

    def get_pixelised_shear(
            self,
            shear_catalog=None,
            SUM=False,
            return_squares=True,
            return_barycenters=False):
        """
        Returns the pixelised shear values, pixel coordinates and optionally their squares and barycenters. If the catalog has not been pixelised yet, it will be pixelised using the provided shear catalog and parameters.
        """

        self.check_pixelisation(
            shear_catalog,
            SUM,
            return_squares,
            return_barycenters)

        # Check if requested parameters are consistent with the current
        # pixelisation. If not, redo pixelisation.
        if (return_barycenters and not self._barycenters) or (
                return_squares and not self._squares):
            if shear_catalog is None:
                raise ValueError(
                    "Barycenters or squares not availiable and no shear catalog provided for pixelisation.")
            self.pixelise_catalog(
                shear_catalog,
                SUM,
                return_squares,
                return_barycenters)

        if return_squares:
            return self.ra, self.dec, self.gamma1, self.gamma2, self.gamma1_sq, self.gamma2_sq
        else:
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

        if self._sum:
            non_zero_weights = self.ng_weight[neighbors][self.ng_weight[neighbors] > 0]
            if len(non_zero_weights) > 0:
                avg_weight = np.mean(non_zero_weights)
                gamma1_j /= avg_weight
                gamma2_j /= avg_weight

                if self._squares:
                    avg_weight_sq = np.mean(non_zero_weights**2)
                    gamma1_sq_j /= avg_weight_sq
                    gamma2_sq_j /= avg_weight_sq

        return center, ra_j, dec_j, gamma1_j, gamma2_j, gamma1_sq_j, gamma2_sq_j

    def initialise_mass_aperture(
            self,
            shear_catalog=None,
            return_squares=True,
            return_barycenters=False):
        self.check_pixelisation(
            shear_catalog,
            self._sum,
            return_squares,
            return_barycenters)
        return {}
