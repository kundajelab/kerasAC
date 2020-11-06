from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'keras Accessibility Models (kerasAC)',
    'download_url': 'https://github.com/kundajelab/kerasAC',
    'version': '2.5.2',
    'packages': ['kerasAC'],
    'setup_requires': [],
    'install_requires': ['tensorflow-gpu>=2.3','pysam', 'tiledb>=0.5.2','psutil','tables','numpy>=1.9', 'keras>=2.4', 'h5py', 'pandas','pybigwig','deeplift','abstention','boto3'],
    'scripts': [],
    'entry_points': {'console_scripts': ['kerasAC_train = kerasAC.train:main',
                                         'kerasAC_predict_hdf5=kerasAC.predict_hdf5:main',
                                         'kerasAC_predict_tdb=kerasAC.predict_tiledb:main',
                                         'kerasAC_calibrate=kerasAC.calibrate:main',
                                         'kerasAC_curves=kerasAC.curves:main',
                                         'kerasAC_score=kerasAC.performance_metrics.performance_metrics:main',
                                         'kerasAC_score_bpnet=kerasAC.performance_metrics.bpnet_performance_metrics:main',
                                         'kerasAC_score_bpnet_legacy=kerasAC.performance_metrics.bpnet_performance_metrics_legacy:main',
                                         'kerasAC_interpret=kerasAC.interpret:main',
                                         'kerasAC_bpnet_shap_wrapper=kerasAC.interpret.bpnet_shap_wrapper:main',
                                         'kerasAC_plot_interpretation=kerasAC.plot_interpretation:main',
                                         'kerasAC_cross_validate=kerasAC.cross_validate:main',
                                         'kerasAC_loss_weights_bpnet=kerasAC.helpers.get_loss_weights_for_bpnet:main',
                                         'kerasAC_bigwigs_from_io=kerasAC.helpers.generate_bigwigs_from_bpnet_io:main',
                                         'kerasAC_aggregate_summaries=kerasAC.helpers.aggregate_summaries:main']},
    'name': 'kerasAC'
}

if __name__== '__main__':
    setup(**config)
