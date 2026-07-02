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
        self._sum = SUM
        self.verbose_print(1, f"Binning catalog : npix = {self._npix}")
        start_time = time()
        pix_ind = hp.ang2pix(
            self.nside,
            shear_catalog.ra,
            shear_catalog.dec,
            nest=False,
            lonlat=True)
        ngal_pix = np.bincount(pix_ind, minlength=self._npix)

        if not SUM:
            self.verbose_print(1, "Shear catalog normalised by pixel weights.")
            ng_weight = self.get_healpix_weights(shear_catalog, pix_ind)
            shear_catalog.normalise_weights(pix_ind, ng_weight)
        elif shear_catalog._normalized:
            raise ValueError("Shear catalog has been normalised by pixel weights. It cannot be used with SUM=True.")

        ra, dec = np.radians(shear_catalog.ra), np.radians(shear_catalog.dec)

        pixel_order = np.argsort(pix_ind)
        offsets = np.r_[0, np.cumsum(ngal_pix), shear_catalog.ngal]

        if SUM:
            all_gals = np.empty((shear_catalog.ngal, 5), dtype=np.float64)
        else:
            all_gals = np.empty((shear_catalog.ngal, 4), dtype=np.float64)

        all_gals[:, 0] = ra[pixel_order]
        all_gals[:, 1] = dec[pixel_order]

        self.mask = (ngal_pix > 0)

        all_gals[:, 2] = shear_catalog.gamma1[pixel_order] * \
            shear_catalog.weight[pixel_order]
        all_gals[:, 3] = shear_catalog.gamma2[pixel_order] * \
            shear_catalog.weight[pixel_order]

        if SUM:
            all_gals[:, 4] = shear_catalog.weight[pixel_order]

        self.verbose_print(
            2, f"Catalog binned in {self.time_to_string(time() - start_time)}")
        return all_gals, offsets

    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        all_gals = kwargs.get('all_gals')
        offsets = kwargs.get('offsets')
        ra = kwargs.get('ra')
        dec = kwargs.get('dec')
        neighbors = hp.query_disc(self.nside, vecs[:, i], radius_rad)

        neighbor_lenghts = offsets[neighbors + 1] - offsets[neighbors]
        if self._sum:
            neighbor_data = np.empty(
                (sum(neighbor_lenghts), 5), dtype=all_gals.dtype)
        else:
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
        gamma1_j_sq = None
        gamma2_j_sq = None

        if self._squares:
            gamma1_j_sq = gamma1_j**2
            gamma2_j_sq = gamma2_j**2

        avg_weight, avg_weight_sq = 1, 1
        if self._sum:
            weights_j = neighbor_data[:, 4]
            # the aperture size is mesured in non-empty pixels
            apr_size_pix = np.sum(neighbor_lenghts > 0)
            avg_weight = np.sum(weights_j) / apr_size_pix
            if self._squares:
                avg_weight_sq = np.sum(weights_j**2) / apr_size_pix
            
        center = (ra[i], dec[i])
        return center, ra_j, dec_j, gamma1_j, gamma2_j, gamma1_j_sq, gamma2_j_sq, avg_weight, avg_weight_sq

    def initialise_mass_aperture(
            self,
            shear_catalog=None,
            return_squares=True,
            return_barycenters=False):
        ra, dec = self.get_healpix_ra_dec()
        all_gals, offsets = self.bin_catalog(shear_catalog, SUM=self._sum)

        return {"all_gals": all_gals, "offsets": offsets, "ra": ra, "dec": dec}
