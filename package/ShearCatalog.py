import numpy as np

class ShearCatalog():
    def __init__(self, shear_catalog, columns=["RA", "DEC", "GAMMA1", "GAMMA2", "WEIGHT", "Z"], gamma1_sign=1, gamma2_sign=1):
        self.shear_catalog = shear_catalog
        self.ra = shear_catalog[columns[0]]
        self.dec = shear_catalog[columns[1]]
        self.gamma1 = shear_catalog[columns[2]].astype(np.float64) * gamma1_sign
        self.gamma2 = shear_catalog[columns[3]].astype(np.float64) * gamma2_sign
        self.weight = shear_catalog[columns[4]].astype(np.float64)
        if len(columns) > 5:
            self.phz = shear_catalog[columns[5]].astype(np.float64)
        else:
            self.phz = None
        self.ngal = np.shape(self.ra)[0]
        
    def filter_by_z(self, z_cut):
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
    
    def normalise_weights(self, npix, npix_ind, ng, SUM=False):
        ng_weight = np.zeros(npix, dtype=np.float64)
        np.add.at(ng_weight, npix_ind, self.weight)
        if SUM:
            ng_weight = np.mean(ng_weight[ng > 0])
        else:
            ng_weight[ng_weight == 0] = 1
        self.weight /= ng_weight[npix_ind]