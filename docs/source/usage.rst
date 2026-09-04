Usage
=====

.. code-block:: python

   from smart import ShearCatalog, PixelMassAperture
   from astropy.io import fits

   data = fits.getdata("catalogue.fits")
   shear_catalog = ShearCatalog(data)
   mass_aperture = PixelMassAperture(nside=2048)
   mapE, mapB, map_vnoise, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog)

A more complete example is provided in ``test.py`` at the root of the repository.
