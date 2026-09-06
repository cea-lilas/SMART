Usage
=====

.. code-block:: python

   from smart import *
   from astropy.io import fits
   import healpy as hp
   import matplotlib.pyplot as plt


Catalog loading
###############

You can load a galaxy catalogue from a FITS file using the `ShearCatalog` class. The catalogue is expected to contain columns for right ascension, declination, shear components (gamma1, gamma2), and weight. You need to specify the column names in the order they appear in the catalogue.

.. code-block:: python

   # Read the galaxy catalogue that is expected to have the following columns:
   # right ascension, declination, shear components (gamma1, gamma2), and weight.
   data = fits.getdata("hsc_catalogue_filtered.fits")

   # Define the columns to be read from the catalogue (in the order: RA, DEC, gamma1, gamma2, weight)
   columns = ["SHE_RA", "SHE_DEC", "SHE_E1_CORRECTED", "SHE_E2_CORRECTED", "SHE_WEIGHT"]
   shear_catalog = ShearCatalog(data, columns=columns, gamma2_sign=1)


Mass aperture implementations
#############################

Four different implementations of the mass aperture computation are available, each with its own advantages and trade-offs in terms of precision and computation time. The following examples demonstrate how to use each implementation.

Galaxy position-based (brute force)
***********************************

.. code-block:: python

   mass_aperture = FullCatMassAperture(nside=2048, verbosity=2)
   mapE, mapB, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=False)


Pixel-based (HEALPix centres)
*****************************

.. code-block:: python

   mass_aperture = PixelMassAperture(nside=2048, verbosity=2)
   mapE, mapB, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=False)


Pixel-based (galaxy position barycentres)
*****************************************

.. code-block:: python

   mass_aperture = PixelMassAperture(nside=2048, verbosity=2)
   mapE, mapB, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=False, barycenters=True)


Galaxy position-based (HEALPix catalogue partitioning)
******************************************************

.. code-block:: python

   mass_aperture = BinCatMassAperture(nside=2048, verbosity=2)
   mapE, mapB, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=False)


SUM vs MEAN approach
####################

Independently of the implementation, the mass aperture can be computed using either a SUM or a MEAN approach (see paper, Sect 2.4).

By default, the MEAN approach is used. To use the SUM approach, you can set the ``SUM`` parameter to ``True`` when calling the :meth:`get_mass_aperture <smart.MassAperture.MassAperture.MassApertureMap.get_mass_aperture>` method.

.. code-block:: python

   mapE, mapB, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=False, SUM=True)


Returning the noise map
#######################

The noise map can be returned by setting the ``return_noise`` parameter to ``True`` when calling the :meth:`get_mass_aperture <smart.MassAperture.MassAperture.MassApertureMap.get_mass_aperture>` method.

.. code-block:: python

   mapE, mapB, map_vnoise, mask = mass_aperture.get_mass_aperture(r_theta_cut=3, shear_catalog=shear_catalog, return_noise=True)
