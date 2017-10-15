#
# h5cc compiles for linking with HDF5 library
#
CC = h5pcc 
CFLAGS =  -O3 -fopenmp -I/usr/include -Wall -Werror -I/home/brryan/Software/gsl/include -std=c99
LDFLAGS = -L/home/brryan/Software/gsl/lib -lm -lgsl -lgslcblas 

# MODELS: harm3d bhlight2d bhlight3d
MODEL = bhlight3d

SRCIPO = \
geodesics.c \
image.c \
main.c radiation.c tetrads.c ipolarray.c \
model_tetrads.c model_radiation.c \
model_geodesics.c \
model_$(MODEL).c \
geometry.c
# model_geometry.c
# geodesics_gsl.c

OBJIPO = \
geodesics.o \
image.o \
main.o radiation.o tetrads.o ipolarray.o \
model_tetrads.o model_radiation.o model_geodesics.o \
model_$(MODEL).o \
geometry.o
# model_geometry.o
# geodesics_gsl.o


ipole: $(OBJIPO) makefile 
	$(CC) $(CFLAGS) -o ipole $(OBJIPO) $(LDFLAGS)

$(OBJIPO) : makefile decs.h defs.h constants.h

clean:
	rm *.o 
cleanup:
	rm ipole*.ppm ipole.dat



