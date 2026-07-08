import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from .MassAperture import MassApertureMap
from scipy.spatial import cKDTree

class FullCatMassAperture(MassApertureMap):

    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        shear_catalog : ShearCatalog = kwargs.get('shear_catalog')
        g1 = kwargs.get('g1')
        g2 = kwargs.get('g2')
        galaxy_KDtree = kwargs.get('kdtree')
        center = np.radians(hp.pix2ang(self.nside, i, lonlat=True))

        neighbors = galaxy_KDtree.query_ball_point(
            vecs[:, i], 2 * np.sin(radius_rad / 2))

        g1_j = g1[neighbors]
        g2_j = g2[neighbors]
        ra_j = np.radians(shear_catalog.ra[neighbors])
        dec_j = np.radians(shear_catalog.dec[neighbors])

        g1_j_sq = None
        g2_j_sq = None

        if self._squares:
            g1_j_sq = g1_j**2
            g2_j_sq = g2_j**2

        avg_weight = 1
        if self._sum and len(neighbors) > 0:
            apt_size_pix = np.sum(self.mask[hp.query_disc(self.nside, vecs[:, i], radius_rad)])
            if apt_size_pix > 0:
                avg_weight = np.sum(shear_catalog.weight[neighbors])/apt_size_pix

        return center, ra_j, dec_j, g1_j, g2_j, g1_j_sq, g2_j_sq, avg_weight

    def initialise_mass_aperture(
            self,
            shear_catalog: ShearCatalog = None,
            return_squares=False,
            return_barycenters=False):
        pix_ind = hp.ang2pix(
            self.nside,
            shear_catalog.ra,
            shear_catalog.dec,
            lonlat=True)
        ng_weight = self.get_healpix_weights(shear_catalog, pix_ind)

        self.mask = ng_weight > 0

        if not self._sum:
            self.verbose_print(1, "Shear catalog normalised by pixel weights.")
            shear_catalog.normalise_weights(pix_ind, ng_weight)
        elif shear_catalog._normalized:
            raise ValueError("Shear catalog has been normalised by pixel weights. It cannot be used with SUM=True.")

        g1 = shear_catalog.gamma1 * shear_catalog.weight
        g2 = shear_catalog.gamma2 * shear_catalog.weight

        vec = hp.ang2vec(shear_catalog.ra, shear_catalog.dec, lonlat=True)

        galaxy_KDtree = cKDTree(vec)
        return {
            "kdtree": galaxy_KDtree,
            "shear_catalog": shear_catalog,
            "g1": g1,
            "g2": g2}
