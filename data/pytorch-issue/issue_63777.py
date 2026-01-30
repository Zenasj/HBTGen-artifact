#12 [conda-installs 1/2] RUN /opt/conda/bin/conda install -c "pytorch-nightly" -c "nvidia" -y python=3.7 pytorch torchvision torchtext "cudatoolkit=11.1" &&     /opt/conda/bin/conda clean -ya

#12 271.7 UnsatisfiableError: The following specifications were found to be incompatible with a past
#12 271.7 explicit spec that is not an explicit spec in this operation (python):
#12 271.7 
#12 271.7   - python=3.7
#12 271.7   - pytorch -> dataclasses -> python[version='>=3.5|>=3.6,<3.7|>=3.7']
#12 271.7   - pytorch -> numpy[version='>=1.11.3,<2.0a0|>=1.11|>=1.11.*|>=1.19|>=1.16.6,<2.0a0|>=1.14.6,<2.0a0|>=1.9.3,<2.0a0|>=1.9']
#12 271.7   - pytorch -> python[version='>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.8,<3.9.0a0|>=3.9,<3.10.0a0|>=3.7,<3.8.0a0|>=3.5,<3.6.0a0']
#12 271.7   - torchtext -> python[version='>=3.6,<3.7.0a0|>=3.9,<3.10.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0']
#12 271.7   - torchtext -> requests -> python[version='>=2.7|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']
#12 271.7   - torchvision -> numpy[version='>1.11|>=1.11|>=1.20']
#12 271.7   - torchvision -> python[version='>=2.7,<2.8.0a0|>=3.6,<3.7.0a0|>=3.7,<3.8.0a0|>=3.8,<3.9.0a0|>=3.9,<3.10.0a0|>=3.5,<3.6.0a0']
#12 271.7   - torchvision -> pytorch==1.9.0.dev20210415 -> numpy[version='>=1.11.*|>=1.19|>=1.16.6,<2.0a0|>=1.11.3,<2.0a0|>=1.14.6,<2.0a0|>=1.9.3,<2.0a0|>=1.9']
#12 271.7   - torchvision -> six -> python

#12 271.7   - torchtext -> requests -> python[version='>=2.7|>=2.7,<2.8.0a0|>=3.5,<3.6.0a0']