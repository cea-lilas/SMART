import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from .MassAperture import MassApertureMap
from time import time


class BinCatMassAperture(MassApertureMap):
    """
    Mass aperture class that groups the shear catalog into pixels while maintaining individual galaxy information for each pixel. This allows for faster retrieval of neighboring galaxies during the mass aperture computation.
    """

    def bin_catalog(self, shear_catalog: ShearCatalog, SUM=False):
        """
        Groups the shear catalog into pixels in oreder to speed up retrieval.

        shear_catalog : ShearCatalog
            A ShearCatalog object containing the galaxy shear data to be binned.
        SUM : Boolean, optional
            If False, normalises the shear weights per pixel, otherwise normalises by the average over the whole map. Defaults to False.
        """
        self.verbose_print(1, f"Binning catalog : npix = {self._npix}")
        start_time = time()
        pix_ind = hp.ang2pix(
            self.nside,
            shear_catalog.ra,
            shear_catalog.dec,
            nest=False,
            lonlat=True)
        ngal_pix = np.bincount(pix_ind, minlength=self._npix)
        shear_catalog.normalise_weights(self._npix, pix_ind, ngal_pix, SUM=SUM)

        ra, dec = np.radians(shear_catalog.ra), np.radians(shear_catalog.dec)

        pixel_order = np.argsort(pix_ind)
        offsets = np.r_[0, np.cumsum(ngal_pix), shear_catalog.ngal]

        all_gals = np.empty((shear_catalog.ngal, 4), dtype=np.float64)

        all_gals[:, 0] = ra[pixel_order]
        all_gals[:, 1] = dec[pixel_order]

        self.mask = (ngal_pix > 0)

        all_gals[:, 2] = shear_catalog.gamma1[pixel_order] * \
            shear_catalog.weight[pixel_order]
        all_gals[:, 3] = shear_catalog.gamma2[pixel_order] * \
            shear_catalog.weight[pixel_order]
        self.verbose_print(
            2, f"Catalog binned in {
                time() - start_time:.2f} seconds.")
        return all_gals, offsets

    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        all_gals = kwargs.get('all_gals')
        offsets = kwargs.get('offsets')
        ra = kwargs.get('ra')
        dec = kwargs.get('dec')
        neighbors = hp.query_disc(self.nside, vecs[:, i], radius_rad)

        neighbor_lenghts = offsets[neighbors + 1] - offsets[neighbors]
        neighbor_data = np.empty(
            (sum(neighbor_lenghts), 4), dtype=all_gals.dtype)

        pos = 0
        for n in range(len(neighbors)):
            neighbor_data[pos: pos + neighbor_lenghts[n],
                          :] = all_gals[offsets[neighbors[n]]: offsets[neighbors[n] + 1], :]
            pos += neighbor_lenghts[n]

        ra_j = neighbor_data[:, 0]
        dec_j = neighbor_data[:, 1]
        gamma1_j = neighbor_data[:, 2]
        gamma2_j = neighbor_data[:, 3]
        if self._squares:
            gamma1_j_sq = gamma1_j**2
            gamma2_j_sq = gamma2_j**2
        else:
            gamma1_j_sq = None
            gamma2_j_sq = None
        center = (ra[i], dec[i])
        return center, ra_j, dec_j, gamma1_j, gamma2_j, gamma1_j_sq, gamma2_j_sq

    def initialise_mass_aperture(
            self,
            shear_catalog=None,
            SUM=False,
            return_squares=True,
            return_barycenters=False):
        ra, dec = self.get_healpix_ra_dec()
        all_gals, offsets = self.bin_catalog(shear_catalog)

        return {"all_gals": all_gals, "offsets": offsets, "ra": ra, "dec": dec}
