import numpy as np
import healpy as hp
from ShearCatalog import ShearCatalog
from abc import ABC, abstractmethod

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
    def __init__(self, nside=2048):
        self.nside = nside
        self._npix = hp.nside2npix(self.nside)
        self._psize_rad = hp.nside2resol(self.nside)
    
    @staticmethod
    def get_jarvis(theta):
        return lambda r: (r**2 / (4 * np.pi *(theta)**4)) * np.exp(-r**2 / (2. * (theta)**2))

    
    def expand_mask(self, mask, radius_rad, vecs):
        mask_hp = mask.copy()
        masked_pixels = np.nonzero(self.mask)[0]
        print('Number of masked pixels =', len(masked_pixels))
        for pix in masked_pixels:
            if not np.all(mask_hp[hp.get_all_neighbours(self.nside, pix)]):
                disc = hp.query_disc(self.nside, vecs[:, pix], radius_rad)
                mask_hp[disc] = True
        print('Number of masked pixels =', np.sum(mask_hp))
        ind = np.nonzero(mask_hp)[0]
        return mask_hp, ind

    def apply_filter(self, filter, center, gamma1, gamma2, ra, dec, gamma1_sq=None, gamma2_sq=None):
        
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
        
        if gamma1_sq is None or gamma2_sq is None:
            return mapE_i, mapB_i
        
        map_vnoise_i = 1/2* np.sum((gamma1_sq + gamma2_sq) * Q**2)
        return mapE_i, mapB_i, map_vnoise_i

    def query_neighbors(self, neighbors):
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
            
            return ra_j, dec_j, gamma1_j, gamma2_j, gamma1_sq_j, gamma2_sq_j

    def get_healpix_ra_dec(self):
        theta, ra = hp.pix2ang(self.nside, np.arange(self._npix), lonlat=False)
        dec = np.pi/2 - theta
        return ra, dec

    def get_healpix_vec(self):
        return np.array(hp.pix2vec(self.nside, np.arange(self._npix)))
    
    @abstractmethod
    def get_mass_aperture(self, filter, r_theta_cut):
        pass