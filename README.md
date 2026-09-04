This package allows a fast and precise computation of spherical aperture mass maps from the shear field, defined directly on the celestial sphere. The method SMART does not rely on planar projections and does not require the reconstruction of convergence maps.

The package offers 3 classes for 4 methods of computing aperture mass, ranging in precision and computation time.

### SMART team (CEA Paris-Saclay)

* Igor Bezmaternykh
* Sandrine Pires
* Nicolas Dagoneau
* Gabriel W. Pratt
* Loris Chappuis
* Gavin Leroy

### Installation

```bash
git clone https://github.com/Igor-Bzk/SMART
cd SMART
pip install .
```

### Example usage

```python
from smart import ShearCatalog, PixelMassAperture
from astropy.io import fits

data = fits.getdata("catalogue.fits")
shear_catalog = ShearCatalog(data)
mass_aperture = PixelMassAperture(nside=2048)
mapE, mapB, map_vnoise, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog)
```

A more complete example is provided in `test.py`

### Acknowledgment

If you use the code SMART in any resulting work, we kindly ask you that you cite the following paper: Bezmathernykh et al. 2026  [link]. For any questions and feedback send an email to [Sandrine Pires](mailto:sandrine.pires@cea.fr).

