kerasAC_calibrate --model DNASE.K562.regressionlabels.withgc.0.hdf5 \
		  --preacts gc.dnase.k562.predictions.preacts.0 \
		  --labels gc.dnase.k562.labels.0 \
		  --outf gc.dnase.k562.predictions.calibrated.0 \
		  --calibrate_regression
