from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'keras Accessibility Models (kerasAC)',
    'download_url': 'https://github.com/kundajelab/kerasAC',
    'version': '2.3',
    'packages': ['kerasAC'],
    'setup_requires': [],
    'install_requires': ['pysam', 'tiledb>=0.5.2','psutil','tables','numpy>=1.9', 'keras>=2.2', 'h5py', 'pandas','pybigwig','deeplift','abstention','boto3'],
    'scripts': [],
    'entry_points': {'console_scripts': ['kerasAC_train = kerasAC.train:main',
                                         'kerasAC_predict_hdf5=kerasAC.predict_hdf5:main',
                                         'kerasAC_predict_tdb=kerasAC.predict_tiledb:main',
                                         'kerasAC_calibrate=kerasAC.calibrate:main',
                                         'kerasAC_curves=kerasAC.curves:main',
                                         'kerasAC_score=kerasAC.performance_metrics.performance_metrics:main',
                                         'kerasAC_score_bpnet=kerasAC.performance_metrics.bpnet_performance_metrics:main',
                                         'kerasAC_interpret=kerasAC.interpret:main',
                                         'kerasAC_plot_interpretation=kerasAC.plot_interpretation:main',
                                         'kerasAC_cross_validate=kerasAC.cross_validate:main',
                                         'kerasAC_loss_weights_bpnet=kerasAC.helpers.get_loss_weights_for_bpnet:main']},
    'name': 'kerasAC'
}

if __name__== '__main__':
    setup(**config)
