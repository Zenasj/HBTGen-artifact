if IS_WINDOWS:
	nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
else:
	nvcc_gendeps = '--generate-nonsystem-dependencies-with-compile --dependency-output $out.d'