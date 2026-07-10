This package allows to compute aperture mass maps from shear catalogs.

The package offers 3 classes for 4 methods of computing aperture mass, ranging in precision and computation time.

For more information, see:

```bibtex
@article{bezmaternykh2026smart,
  title={SMART: Spherical Mass ApeRture Toolkit},
  author={}
  journal={},
  year={2026},
  doi={}
}
```

### Installation:

```bash
git clone https://github.com/Igor-Bzk/SMART
cd SMART
pip install .
```

### Example usage:

```python
from smart import ShearCatalog, PixelMassAperture
from astropy.io import fits

data = fits.getdata("catalogue.fits")
shear_catalog = ShearCatalog(data)
mass_aperture = PixelMassAperture(nside=2048)
mapE, mapB, map_vnoise, mask = mass_aperture.get_mass_aperture(3, shear_catalog=shear_catalog)
```

A more complete example is provided in `test.py`
