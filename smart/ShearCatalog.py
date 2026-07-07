import numpy as np
from astropy.table import Table

class ShearCatalog():
    """
    Class to handle galaxy shear catalogs
    """

    def __init__(
            self,
            shear_catalog,
            columns=[
                "RA",
                "DEC",
                "GAMMA1",
                "GAMMA2",
                "WEIGHT",
                "Z"],
            gamma1_sign=1,
            gamma2_sign=1):
        """
        Initializes the ShearCatalog object by extracting the relevant columns from the input catalog and applying the specified signs to the shear components.
        Args:
            shear_catalog : record array or table-like
                Input catalog containing the galaxy shear data.
            columns : list of str, optional
                List of column names in the input catalog corresponding to RA, Dec, gamma1, gamma2, weight, and optionally photometric redshift. Defaults to ["RA", "DEC", "GAMMA1", "GAMMA2", "WEIGHT", "Z"].
            gamma1_sign : int, optional
                Sign to apply to the gamma1 component (1 or -1). Defaults to 1.
            gamma2_sign : int, optional
                Sign to apply to the gamma2 component (1 or -1). Defaults to 1.
        """
        if shear_catalog.dtype.byteorder not in ('=', '|'):
            # Fits files are big-endian, ensure the data is in native byte order for processing
            shear_catalog.byteswap().view(shear_catalog.dtype.newbyteorder('='))
        self._normalized = False
        self.ra = shear_catalog[columns[0]]
        self.dec = shear_catalog[columns[1]]
        self.gamma1 = shear_catalog[columns[2]] * gamma1_sign
        self.gamma2 = shear_catalog[columns[3]] * gamma2_sign
        self.weight = shear_catalog[columns[4]]
        if len(columns) > 5:
            self.phz = shear_catalog[columns[5]]
        else:
            self.phz = None
        self.ngal = np.shape(self.ra)[0]

    def filter_by_z(self, z_cut):
        """
        Filters the catalog by photometric redshift, keeping only galaxies with z > z_cut.

        Args:
            z_cut : float
                Redshift cut to apply to the catalog.
        """
        if self.phz is None:
            raise ValueError("No photometric redshift column is specified.")
        in_phz = self.in_data[self.phz]
        z_mask = in_phz > z_cut
        self.ra = self.ra[z_mask]
        self.dec = self.dec[z_mask]
        self.g1 = self.g1[z_mask]
        self.g2 = self.g2[z_mask]
        self.weight = self.weight[z_mask]
        self.ngal = np.shape(self.ra)[0]
        return self

    def normalise_weights(self, pix_ind, ng_weight):
        """
        Normalises the shear weights of the catalog by pixel.

        Args:
            npix : int
                Total number of pixels in the map.
            pix_ind : np.ndarray
                Array of pixel indices corresponding to each galaxy in the catalog.
        """
        if self._normalized:
            print("Weights are already normalised.")
            return
        ng_weight[ng_weight == 0] = 1
        self.weight /= ng_weight[pix_ind]
        self._normalized = True
    
    def save_to_fits(self, filename):
        """
        Saves the shear catalog to a FITS file.

        Args:
            filename : str
                Name of the output FITS file.
        """
        table = Table()
        table['RA'] = self.ra
        table['DEC'] = self.dec
        table['GAMMA1'] = self.gamma1
        table['GAMMA2'] = self.gamma2
        table['WEIGHT'] = self.weight
        if self.phz is not None:
            table['Z'] = self.phz
        table.write(filename, format='fits', overwrite=True)