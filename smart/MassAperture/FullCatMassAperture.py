import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from .MassAperture import MassApertureMap
from scipy.spatial import cKDTree


class FullCatMassAperture(MassApertureMap):

    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        shear_catalog = kwargs.get('shear_catalog')
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

        return center, ra_j, dec_j, g1_j, g2_j, g1_j_sq, g2_j_sq

    def initialise_mass_aperture(
            self,
            shear_catalog: ShearCatalog = None,
            SUM=False,
            return_squares=False,
            return_barycenters=False):
        pix_ind = hp.ang2pix(
            self.nside,
            shear_catalog.ra,
            shear_catalog.dec,
            lonlat=True)
        self.verbose_print(2, "Applying shear weights...")
        ng = np.bincount(pix_ind, minlength=self._npix)
        shear_catalog.normalise_weights(self._npix, pix_ind, ng, SUM=False)
        g1 = shear_catalog.gamma1 * shear_catalog.weight
        g2 = shear_catalog.gamma2 * shear_catalog.weight

        self.mask = ng > 0

        vec = hp.ang2vec(shear_catalog.ra, shear_catalog.dec, lonlat=True)

        galaxy_KDtree = cKDTree(vec)
        return {
            "kdtree": galaxy_KDtree,
            "shear_catalog": shear_catalog,
            "g1": g1,
            "g2": g2}
