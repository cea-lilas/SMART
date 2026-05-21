import numpy as np
import healpy as hp
from .ShearCatalog import ShearCatalog
from .filters import *
from abc import ABC, abstractmethod
from astropy.table import Table
from time import time

def polar_angle(dec_i, ra_i, dec_j, ra_j):
    """
    Gives the direction to go from first galaxy to second galaxy in the sphere (measured from North)
    Inputs are longitude and latitude
    (dec_i, ra_i) vers (dec_j, ra_j).

    dec_i, ra_i : longitude and latitude of i [in rad] - float ou np.ndarray
    dec_j, ra_j : longitude and latitude of j [in rad] - float ou np.ndarray
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
    def __init__(self, nside=2048, verbosity=1):
        self.nside = nside
        self._npix = hp.nside2npix(self.nside)
        self._psize_rad = hp.nside2resol(self.nside)
        self._squares = False
        self.verbosity = verbosity
    
    def verbose_print(self, verbosity, *args):
        if self.verbosity >= verbosity:
            print(*args)

    def get_filter_by_name(self, name, theta=None):
        match name:
            case "jarvis":
                if theta is None:
                    theta = 2.3*(self.nside//2048)
                return get_jarvis(theta)
            case _:
                raise ValueError(f"Filter {name} not found. Available filters: 'jarvis'.")
    
    def expand_mask(self, radius_rad, vecs):
        mask_hp = self.mask.copy()
        masked_pixels = np.nonzero(self.mask)[0]
        self.verbose_print(1, f'Number of masked pixels = {len(masked_pixels)}')
        for pix in masked_pixels:
            if not np.all(mask_hp[hp.get_all_neighbours(self.nside, pix)]):
                disc = hp.query_disc(self.nside, vecs[:, pix], radius_rad)
                mask_hp[disc] = True
        ind = np.nonzero(mask_hp)[0]
        self.verbose_print(1, f"Masked pixels after expansion = {len(ind)}")
        return mask_hp, ind

    def apply_filter(self, filter, i, vecs, radius_rad, **kwargs):
        center, ra, dec, gamma1, gamma2, gamma1_sq, gamma2_sq= self.query_neighbors(i, vecs, radius_rad, **kwargs)
        # print(gamma1)
        center_ra, center_dec = center
        
        r = np.arccos(np.sin(dec)*np.sin(center_dec) + np.cos(dec)*np.cos(center_dec) *np.cos(ra-center_ra))
        r /= self._psize_rad

        # Derive the angle between the two segment (using longitude and latitude)
        psi = polar_angle(center_dec, center_ra, dec, ra)

        # Compute the tangential and cross shear
        gamma_t =   gamma1 * np.cos(2 * psi) - gamma2 * np.sin(2 * psi)
        gamma_x =  -gamma1 * np.sin(2 * psi) - gamma2 * np.cos(2 * psi)
        # Aperture mass filter
        Q = filter(r)
        mapE_i = np.sum(gamma_t * Q)
        mapB_i = np.sum(gamma_x * Q)
        if not self._squares:
            return mapE_i, mapB_i
        
        map_vnoise_i = 1/2* np.sum((gamma1_sq + gamma2_sq) * Q**2)
        return mapE_i, mapB_i, map_vnoise_i

    def get_healpix_ra_dec(self):
        theta, ra = hp.pix2ang(self.nside, np.arange(self._npix), lonlat=False)
        dec = np.pi/2 - theta
        return ra, dec

    def get_healpix_vec(self):
        return np.array(hp.pix2vec(self.nside, np.arange(self._npix)))
    
    @abstractmethod
    def query_neighbors(self, i, vecs, radius_rad, **kwargs):
        pass
    
    @abstractmethod
    def initialise_mass_aperture(self, shear_catalog=None, SUM = False, return_squares=True, return_barycenters=False):
        pass
    
    def get_mass_aperture(self, r_theta_cut, filter="jarvis", shear_catalog : ShearCatalog = None, SUM = False, return_noise=True, return_barycenters=False):
        self.verbose_print(1, f"Number of galaxies = {shear_catalog.ngal}")
        self._squares = return_noise
        kwargs = self.initialise_mass_aperture(shear_catalog, SUM, return_noise, return_barycenters)
        start_time = time()
        self.verbose_print(2, "Starting mass aperture computation...")
        
        if isinstance(filter, str):
            filter = self.get_filter_by_name("jarvis")
        if not callable(filter):
            raise ValueError("Filter must be a valid filter name or a callable function taking one argument r.")
        
        mapE = np.zeros(self._npix)
        mapB = np.zeros(self._npix)
        if self._squares:
            map_vnoise = np.zeros(self._npix)
        
        radius_rad = r_theta_cut * self._psize_rad

        pix_vec = self.get_healpix_vec()
        
        mask_hp, ind = self.expand_mask(radius_rad, pix_vec)
        
        for i in ind:
            if self._squares:
                mapE[i], mapB[i], map_vnoise[i] = self.apply_filter(filter, i, pix_vec, radius_rad, **kwargs)
            else:
                mapE[i], mapB[i] = self.apply_filter(filter, i, pix_vec, radius_rad, **kwargs)
        
        self.verbose_print(2, f"Mass aperture computation done in {time() - start_time:.2f} seconds.")
        
        if self._squares:
            return mapE, mapB, map_vnoise, mask_hp
        else:
            return mapE, mapB, mask_hp
    
    def save_mass_aperture(self, filename, r_theta_cut, filter=None, shear_catalog=None, SUM = False, return_noise=True, return_barycenters=False):
        if return_noise:
            mapE, mapB, map_vnoise, mask_hp = self.get_mass_aperture(r_theta_cut, filter, shear_catalog, SUM, return_noise, return_barycenters)
            Table([mapE, mapB, map_vnoise, mask_hp], names=["mapE", "mapB", "map_vnoise", "mask"]).write(filename, overwrite=True)
        else:
            mapE, mapB, mask_hp = self.get_mass_aperture(r_theta_cut, filter, shear_catalog, SUM, return_noise, return_barycenters)
            Table([mapE, mapB, mask_hp], names=["mapE", "mapB", "mask"]).write(filename, overwrite=True)