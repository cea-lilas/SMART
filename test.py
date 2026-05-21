from package import *
from astropy.io import fits
from os import chdir
import healpy as hp
import matplotlib.pyplot as plt

chdir("/local/home/ib286534/Documents/Catalogues/")

# data = fits.getdata("RR2_filtered.fits")
data = fits.getdata("hsc_catalogue_filtered.fits")

shear_catalog = ShearCatalog(data, columns=["RA", "Dec", "e_1", "e_2", "weight"], gamma2_sign=1)
mass_aperture = PixelMassAperture(nside=2048, verbosity=2)
mapE, mapB, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog, return_noise=False)
hp.mollview(mapE, cmap="gist_stern")
plt.show()
hp.gnomview(mapE, rot=(0, 0), cmap="gist_stern")
# hp.gnomview(mapE, rot=(55, -58), cmap="gist_stern")
plt.show()