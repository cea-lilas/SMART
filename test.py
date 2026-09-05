from smart import *
from astropy.io import fits
from os import chdir
import healpy as hp
import matplotlib.pyplot as plt

# Move to the directory where the galaxy catalogue is located
chdir("/path/to/catalogue/")

# Read the galaxy catalogue that is expected to have the following columns: 
# right ascension, declination, shear components (gamma1, gamma2), and weight.
data = fits.getdata("hsc_catalogue_filtered.fits")

# Define the columns to be read from the catalogue (in the order: RA, DEC, gamma1, gamma2, weight)
columns = ["SHE_RA", "SHE_DEC", "SHE_E1_CORRECTED", "SHE_E2_CORRECTED", "SHE_WEIGHT"]
shear_catalog = ShearCatalog(data, columns=columns, gamma2_sign=1)

# Run the mass aperture defining the ouput resolition (nside)
mass_aperture = PixelMassAperture(nside=2048, verbosity=2)
# or mass_aperture = BinCatMassAperture(nside=2048, verbosity=2)
# or mass_aperture = FullCatMassAperture(nside=2048, verbosity=2)

mapE, mapB, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog, return_noise=False)

# Use barycenters=True with the pixelated method to calculate pixel barycenters and use them as pixel positions
# mapE, mapB, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog, return_noise=False, barycenters=True)

# Choose SUM=True, for the summed approach (default is averaged approach).
# mapE, mapB, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog, return_noise=False, SUM=True)

# With return_noise=True, the function returns the noise map as well:
# mapE, mapB, map_vnoise, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog, return_noise=True)

# Plot the E-mode map using Mollweide and Gnomonic projections
hp.mollview(mapE, cmap="gist_stern")
plt.show()
hp.gnomview(mapE, rot=(0, 0), cmap="gist_stern")
plt.show()
