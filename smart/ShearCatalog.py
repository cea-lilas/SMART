import numpy as np


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
        self.shear_catalog = shear_catalog
        self.ra = shear_catalog[columns[0]]
        self.dec = shear_catalog[columns[1]]
        self.gamma1 = shear_catalog[columns[2]].astype(
            np.float64) * gamma1_sign
        self.gamma2 = shear_catalog[columns[3]].astype(
            np.float64) * gamma2_sign
        self.weight = shear_catalog[columns[4]].astype(np.float64)
        if len(columns) > 5:
            self.phz = shear_catalog[columns[5]].astype(np.float64)
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
        ng_weight[ng_weight == 0] = 1
        self.weight /= ng_weight[pix_ind]