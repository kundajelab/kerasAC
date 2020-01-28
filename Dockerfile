FROM  kundajelab/anaconda-tensorflow-keras:latest
MAINTAINER Anna Shcherbina <annashch@stanford.edu>

#
# Install kerasAC and dependencies
#
RUN apt-get update
RUN apt-get install -y libz-dev liblzma-dev gcc libcurl4 libcurl4-openssl-dev 		
RUN pip install pysam \
		tiledb>=0.5.2 \
		psutil \
		tables \
		numpy>=1.9 \
		h5py \
		pandas \
		deeplift \
		abstention \
		psutil \
		boto3 \
		pyBigWig


WORKDIR /opt
RUN git clone https://github.com/kundajelab/kerasAC.git
WORKDIR /opt/kerasAC
RUN python setup.py build
RUN python setup.py develop

