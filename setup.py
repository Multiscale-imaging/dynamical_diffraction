import setuptools

with open("README.txt", encoding = 'utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'dynamical_diffraction',
    version = '0.0.1',
    author = 'Mads Carlsen',
    author_email = 'madsac@dtu.dk',
    description = 'Library of functions for dynamical diffraction',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/Multiscale-imaging/dynamical_diffraction',
    packages = ['dynamical_diffraction'],
    python_requires = '>=3.6',
)