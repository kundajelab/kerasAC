from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'keras Accessibility Models (kerasAC)',
    'download_url': 'https://github.com/kundajelab/kerasAC',
    'version': '1',
    'packages': ['kerasAC'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras>=2.2', 'h5py'],
    'scripts': [],
    'entry_points': {'console_scripts': ['kerasAC_train = kerasAC.train:main',
                                         'kerasAC_predict=kerasAC.predict:main',
                                         'kerasAC_curves=kerasAC.curves:main',
                                         'kerasAC_interpret=kerasAC.interpret:main']},
    'name': 'kerasAC'
}

if __name__== '__main__':
    setup(**config)
