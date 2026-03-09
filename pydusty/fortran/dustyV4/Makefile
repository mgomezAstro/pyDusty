DATADIR=$(PWD)/

dusty:
	 gfortran -O3 -lgomp -fopenmp -I$(DATADIR) -cpp -DDATADIR_MACRO=\"$(DATADIR)\" -o dusty dusty.f90
