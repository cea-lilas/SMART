import numpy as np
import healpy as hp
from ..ShearCatalog import ShearCatalog
from ..filters import *
from abc import ABC, abstractmethod
from astropy.table import Table
from time import time
from tqdm import tqdm


def polar_angle(dec_i, ra_i, dec_j, ra_j):
    """
    Gives the direction to go from galaxy i to galaxy j on the surface of the sphere (measured from North).
    (dec_i, ra_i) to (dec_j, ra_j).

    Args:
        dec_i, ra_i : float or np.ndarray
            Longitude and latitude of i [in rad]
        dec_j, ra_j : float or np.ndarray
            Longitude and latitude of j [in rad]
    """
    dec_i = np.asarray(dec_i)
    ra_i = np.asarray(ra_i)
    dec_j = np.asarray(dec_j)
    ra_j = np.asarray(ra_j)

    # Longitude difference
    dra = ra_j - ra_i

    # Position angle
    num = np.sin(dra)
    den = np.cos(dec_i) * np.tan(dec_j) - np.sin(dec_i) * np.cos(dra)
    psi = np.arctan2(num, den)

    # Normalisation 0 and 2π to avoid lines of zero
    psi = np.mod(psi, 2 * np.pi)
    return psi


class MassApertureMap(ABC):
    """
    Generic class for mass aperture map computation.
    """

    def __init__(self, nside=2048, verbosity=1):
        """
        Class for mass aperture map computation.

        Args:
            nside : int
                Healpix nside of the output mass aperture map.
            verbosity : int
                Verbosity level (0: no print, 1: important info, 2: more info (computation time)).
        """
        self.nside = nside
        self._npix = hp.nside2npix(self.nside)
        self._psize_rad = hp.nside2resol(self.nside)
        self._squares = False
        self._sum = False
        self.verbosity = verbosity

    def verbose_print(self, verbosity, *args):
        """
        Utility function for printing messages according to the verbosity level.

        Args:
            verbosity : int
                Verbosity level of the message (0: no print, 1: important info, 2: more info).
            *args :
                Arguments to print if the verbosity level is sufficient.
        """
        if self.verbosity >= verbosity:
            print(*args)

    @staticmethod
    def time_to_string(timestamp):

        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds:.2f}s")

        return " ".join(parts)

    def get_filter_by_name(self, name, theta=None, x_c=0.1):
        """
        Retrives an implemented filter function by its name.

        Args:
            name : str
                Name of the filter to retrieve. Implemented filters: 'jarvis'.
            theta : float, optional
                Scale parameter for the filter (if applicable).
        """
        if theta is None:
            theta = 2.3 * (self.nside // 2048)

        match name:
            case "J04":
                return get_jarvis(theta)
            case "tanh":
                return get_tanh(theta, x_c)
            case "M18":
                return get_miyazaki(theta)
            case _:
                raise ValueError(
                    f"Filter {name} not found. Available filters: 'jarvis'.")

    def expand_mask(self, radius_rad, vecs):
        """
        Expands a healpix mask by a circular radius around masked pixels. Utility function to avoid edge effects in the mass aperture computation.

        Args:
            radius_rad : float
                Expansion radius in radians.
            vecs : np.ndarray
                Healpix pixel 3d vector positions as an array of shape (3, npix).

        Returns the expanded mask and the indices of the pixels that are masked after expansion.
        """
        mask_hp = self.mask.copy()
        masked_pixels = np.nonzero(self.mask)[0]
        self.verbose_print(
            2, f'Number of masked pixels = {len(masked_pixels)}')
        for pix in masked_pixels:
            if not np.all(mask_hp[hp.get_all_neighbours(self.nside, pix)]):
                disc = hp.query_disc(self.nside, vecs[:, pix], radius_rad)
                mask_hp[disc] = True
        ind = np.nonzero(mask_hp)[0]
        self.verbose_print(2, f"Masked pixels after expansion = {len(ind)}")
        return mask_hp, ind

    def apply_filter(self, filter, i, vecs, radius_rad, **kwargs):
        """
        Utility function used in the mass aperture computation loop to apply the mass aperture filter at a given pixel.

        Args:
            filter : callable
                Mass aperture filter function taking one argument r (distance to the center of the aperture in units of the pixel size).
            i : int
                Index of the pixel where to apply the filter.
            vecs : np.ndarray
                Healpix pixel 3d vector positions as an array of shape (3, npix).
            radius_rad : float
                Aperture radius in radians.
            **kwargs :
                Additional arguments to pass to the query_neighbors function, depending on the mass aperture implementation.
        """
        center, ra, dec, gamma1, gamma2, gamma1_sq, gamma2_sq, avg_weight = self.query_neighbors(
            i, vecs, radius_rad, **kwargs)

        center_ra, center_dec = center

        r = np.arccos(
            np.sin(dec) *
            np.sin(center_dec) +
            np.cos(dec) *
            np.cos(center_dec) *
            np.cos(
                ra -
                center_ra))
        r /= self._psize_rad

        psi = polar_angle(center_dec, center_ra, dec, ra)

        # Compute the tangential and cross shear
        gamma_t = gamma1 * np.cos(2 * psi) - gamma2 * np.sin(2 * psi)
        gamma_x = -gamma1 * np.sin(2 * psi) - gamma2 * np.cos(2 * psi)

        # Aperture mass filter
        Q = filter(r)
        mapE_i = np.sum(gamma_t * Q)/avg_weight
        mapB_i = np.sum(gamma_x * Q)/avg_weight
        if not self._squares:
            return mapE_i, mapB_i

        map_vnoise_i = 1 / 2 * np.sum((gamma1_sq + gamma2_sq) * Q**2)/avg_weight**2
        return mapE_i, mapB_i, map_vnoise_i

    def get_healpix_ra_dec(self):
        """
        Utility function to get the longitude and latitude of the center of each healpix pixel.
        """
        theta, ra = hp.pix2ang(self.nside, np.arange(self._npix), lonlat=False)
        dec = np.pi / 2 - theta
        return ra, dec

    def get_healpix_vec(self):
        """
        Utility function to get the 3d vector position of the center of each healpix pixel.
        """
        return np.array(hp.pix2vec(self.nside, np.arange(self._npix)))

    def get_healpix_weights(self, shear_catalog : ShearCatalog, pix_ind):
        """
        Utility function to compute the sum of shear weights in each pixel.

        Args:
            shear_catalog : ShearCatalog
                A ShearCatalog object containing the galaxy shear data.
            pix_ind : np.ndarray
                Array of pixel indices corresponding to each galaxy in the catalog.
        """
        ng_weight = np.bincount(pix_ind, shear_catalog.weight, minlength=self._npix)
        return ng_weight

    @abstractmethod
    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        """
        Abstract method to query the shear values of the galaxies in the aperture of a given pixel. To be implemented in each mass aperture class depending on the data structure used for the shear catalog.

        Args:
            i : int
                Index of the pixel where to apply the filter.
            vecs : np.ndarray
                Healpix pixel 3d vector positions as an array of shape (3, npix).
            radius_rad : float
                Aperture radius in radians.
            **kwargs :
                Additional arguments depending on the mass aperture implementation (e.g. shear catalog, pixelised shear values, etc.).
        """
        pass

    @abstractmethod
    def initialise_mass_aperture(
            self,
            shear_catalog=None,
            return_squares=True,
            return_barycenters=False):
        """
        Abstract method to initialise the mass aperture computation. To be implemented in each mass aperture class depending on the data structure used for the shear catalog. This function is called before the mass aperture computation loop and can be used to prepare any data structure needed for the query_neighbors function (e.g. pixelised shear values, KD-tree, etc.).
        """
        pass

    def get_mass_aperture(
            self,
            r_theta_cut,
            filter="J04",
            shear_catalog: ShearCatalog = None,
            SUM=False,
            return_noise=True,
            barycenters=False):
        """
        Main function to compute the healpix mass aperture map.

        r_theta_cut : float
            Aperture radius in units of the pixel size (e.g. r_theta_cut=3 means an aperture radius of 3 times the pixel size).
        filter : str or callable, optional
            Mass aperture filter to apply. Can be a string corresponding to a implemented filter (e.g. 'jarvis') or a callable function taking one argument r (distance to the center of the aperture in units of the pixel size). Defaults to 'jarvis'.
        shear_catalog : ShearCatalog, optional
            Shear catalog containing the galaxy shear data to be used for the mass aperture computation. Required if the mass aperture implementation needs to query the shear values of individual galaxies (e.g. FullCatMassAperture). Not required if the mass aperture implementation uses pixelised shear values (e.g. PixelMassAperture) and the pixelisation has already been done. Defaults to None.
        SUM : bool, optional
            If True, group the pixels by summing galaxy shears, averages them otherwise. In methods that do not pixelise shear, changes wether shear weights are normalized per pixel or by the average overall shear. Defaults to False.
        return_noise : bool, optional
            If True, also compute and return the noise map. Defaults to True.
        barycenters : bool, optional
            If True, use the barycenter of the galaxies in each pixel (only used if shear is pixelised, where it allows better accuracy). Defaults to False.

        Returns the mass aperture E-mode map, B-mode map, noise map (if return_noise=True) and mask of the pixels used for the computation.
        """
        self.verbose_print(1, f"Number of galaxies = {shear_catalog.ngal}")
        self._squares = return_noise
        self._sum = SUM
        start_time = time()
        kwargs = self.initialise_mass_aperture(
            shear_catalog, return_noise, barycenters)
        self.verbose_print(2, f"SUM = {self._sum}, NOISE = {self._squares}")
        self.verbose_print(2, "Starting mass aperture computation...")
        if isinstance(filter, str):
            filter = self.get_filter_by_name(filter)
        if not callable(filter):
            raise ValueError(
                "Filter must be a valid filter name or a callable function taking one argument r.")

        mapE = np.zeros(self._npix)
        mapB = np.zeros(self._npix)
        if self._squares:
            map_vnoise = np.zeros(self._npix)

        radius_rad = r_theta_cut * self._psize_rad

        pix_vec = self.get_healpix_vec()

        mask_hp, ind = self.expand_mask(radius_rad, pix_vec)

        if self.verbosity >= 1:
            ind = tqdm(
                ind,
                desc="Applying filter",
                unit="pixels",
                mininterval=1)

        for i in ind:
            if self._squares:
                mapE[i], mapB[i], map_vnoise[i] = self.apply_filter(
                    filter, i, pix_vec, radius_rad, **kwargs)
            else:
                mapE[i], mapB[i] = self.apply_filter(
                    filter, i, pix_vec, radius_rad, **kwargs)

        self.verbose_print(
            2, f"Total mass aperture computation time : {self.time_to_string(time() - start_time)}")

        if self._squares:
            return mapE, mapB, map_vnoise, mask_hp
        else:
            return mapE, mapB, mask_hp

    def save_mass_aperture(
            self,
            filename,
            r_theta_cut,
            filter="J04",
            shear_catalog=None,
            SUM=False,
            return_noise=True,
            barycenters=False):
        if return_noise:
            mapE, mapB, map_vnoise, mask_hp = self.get_mass_aperture(
                r_theta_cut, filter, shear_catalog, SUM, return_noise, barycenters)
            output_table = Table([mapE, mapB, map_vnoise, mask_hp], names=[
                  "MAP_E", "MAP_B", "VNOISE", "MASK"])
        else:
            mapE, mapB, mask_hp = self.get_mass_aperture(
                r_theta_cut, filter, shear_catalog, SUM, return_noise, barycenters)
            output_table = Table([mapE, mapB, mask_hp], names=["mapE", "mapB", "mask"])

        output_table.meta['PIXTYPE'] = 'HEALPIX'
        output_table.meta['NSIDE'] = self.nside
        output_table.meta['ORDERING'] = 'RING'
        output_table.meta['INDXSCHM'] = 'IMPLICIT'
        output_table.meta['FIRSTPIX'] = 0
        output_table.meta['LASTPIX'] = self._npix - 1

        output_table.write(filename, overwrite=True)
