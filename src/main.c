#include "decs.h"

#include "model.h"
#include "model_geodesics.h"
#include "model_radiation.h"
#include "model_tetrads.h"

#include "radiation.h"
#include "coordinates.h"
#include "tetrads.h"
#include "geometry.h"
#include "geodesics.h"
#include "image.h"
#include "io.h"
#include "ipolarray.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// Some useful blocks of code to re-use
void get_pixel(int i, int j, int nx, int ny, double Xcam[NDIM], Params params,
               double fovx, double fovy, double freq, int only_unpolarized, double scale,
               double *Intensity, double *Is, double *Qs, double *Us, double *Vs,
               double *Tau, double *tauF);
void save_pixel(double *image, double *imageS, double *taus, int i, int j, int nx, int ny, int only_unpol,
                double Intensity, double Is, double Qs, double Us, double Vs,
                double freqcgs, double Tau, double tauF);
void print_image_stats(double *image, double *imageS, int nx, int ny, Params params, double scale);

// global variables. TODO scope into main
static double tf = 0.;

Params params = { 0 };

// Try to keep dynamic image sizes fast
static inline int imgindex(int n, int i, int j, int nx, int ny) {return (n*nx + i)*ny + j;}

int main(int argc, char *argv[]) 
{
  // motd
  fprintf(stderr, "%s. githash: %s\n", VERSION_STRING, xstr(VERSION));
  fprintf(stderr, "notes: %s\n\n", xstr(NOTES));

  // initialization
  double time = omp_get_wtime();

  double tA, tB; // for slow light
  double Xcam[NDIM];
  double freq, scale;
  double DX, DY, fovx, fovy;

  // Options not needed outside this monster function
  int quench_output = 0;
  int only_unpolarized = 0;

#pragma omp parallel
  if (omp_get_thread_num() == 0) {
    fprintf(stderr, "nthreads = %d\n", omp_get_num_threads());
  }

  // load values from parameter file. handle all actual
  // model parameter comprehension in the model/* files
  load_par_from_argv(argc, argv, &params);

  // figure out if we should run in a custom mode
  // TODO handle with other parameters instead of merging
  for (int i=0; i<argc; ++i) {
    if ( strcmp(argv[i], "-quench") == 0 ) quench_output = 1;
    else if ( strcmp(argv[i], "-unpol") == 0 ) only_unpolarized = 1;
  }
  if (params.quench_output) quench_output = 1;
  if (params.only_unpolarized) only_unpolarized = 1;

  // now that we've loaded all parameters, tell our model about
  // them and use init_model to load the first dump
  init_model(&tA, &tB);

  // Adaptive resolution option
  // nx, ny are the resolution at maximum refinement level
  // nx_min, ny_min are at coarsest level
  // TODO check for obvious BS here
  int nx = params.nx;
  int ny = params.ny;
  if (params.nx_min < 0) {
    params.nx_min = params.nx;
    params.ny_min = params.ny;
  }
  int refine_level = log2(params.nx/params.nx_min)+1;

  // normalize frequency to electron rest-mass energy
  double freqcgs = params.freqcgs;
  freq = params.freqcgs * HPL / (ME * CL * CL);

  // Initialize the camera
  params.rotcam *= M_PI/180.;

  // translate to geodesic coordinates
  native_coord(params.rcam, params.thetacam, params.phicam, Xcam);  // TODO cartesian version
  fprintf(stderr, "Xcam[] = %e %e %e %e\n", Xcam[0], Xcam[1], Xcam[2], Xcam[3]);
  fprintf(stderr, "a=%g R0=%g hslope=%g\n", a, R0, hslope);

  params.dsource *= PC;
  double Dsource = params.dsource; // Shorthand

  // set DX/DY using fov_dsource if possible, otherwise DX, otherwise old default
  double fov_to_d = Dsource / L_unit / 2.06265e11;

  if (params.fovx_dsource != 0.0) { // FOV was specified
    // Uncomment to be even more option lenient
    //if (params.fovy_dsource == 0.0) params.fovy_dsource = params.fovx_dsource;
  } else if (params.dx != 0.0) {
    //if (params.dy == 0.0) params.dy = params.dx;
    params.fovx_dsource = params.dx / fov_to_d;
    params.fovy_dsource = params.dy / fov_to_d;
  } else {
    fprintf(stderr, "No FOV was specified. Using default 160muas!\n");
    params.fovx_dsource = 160.;
    params.fovy_dsource = 160.;
  }
  DX = params.fovx_dsource * fov_to_d;
  DY = params.fovy_dsource * fov_to_d;
  params.dx = DX;
  params.dy = DY;

  // Set the *camera* fov values
  // We don't set these like other parameters, but output them for historical reasons
  fovx = DX / params.rcam;
  fovy = DY / params.rcam;

  scale = (DX * L_unit / nx) * (DY * L_unit / ny) / (Dsource * Dsource) / JY;
  fprintf(stderr,"L_unit = %e DX = %e NX = %i Dsource = %e JY = %e\n", L_unit, DX, nx, Dsource,JY);
  fprintf(stderr,"intensity [cgs] to flux per pixel [Jy] conversion: %g\n",scale);
  fprintf(stderr,"Dsource: %g [cm]\n",Dsource);
  fprintf(stderr,"Dsource: %g [kpc]\n",Dsource/(1.e3*PC));
  fprintf(stderr,"FOVx, FOVy: %g %g [GM/c^2]\n",DX,DY);
  fprintf(stderr,"FOVx, FOVy: %g %g [rad]\n",DX*L_unit/Dsource,DY*L_unit/Dsource);
  fprintf(stderr,"FOVx, FOVy: %g %g [muas]\n",DX*L_unit/Dsource * 2.06265e11 ,DY*L_unit/Dsource * 2.06265e11);
  fprintf(stderr,"Resolution: %dx%d, refined up to %dx%d (%d levels)\n",
          params.nx_min, params.ny_min, params.nx, params.ny, refine_level);
  if (refine_level > 1) {
    fprintf(stderr,"Refinement only when > %f relative error and pixel brightness > %f of average\n",
            params.refine_rel, params.refine_cut);
  }

  double *taus = calloc(nx*ny, sizeof(*taus));
  double *imageS = calloc(nx*ny*NIMG, sizeof(*imageS));
  double *image = calloc(nx*ny, sizeof(*image));

  // slow light
  if (SLOW_LIGHT) {

    // used to track when to write restart files
    double next_restart_after = -1.;

    // maximum number of steps, needed for dtraj (saving geodesics) 
    int maxsteplength = -1;

    // minimum time, used to track the longest geodesic
    double t0 = 0.;

    // time when first geodesic first goes inside of R<100 ball.
    // the difference (t0 - tgeoi) is the minimum amount of time
    // required to be within the dumps. i.e., if the integration
    // must go beyond this range, it is not required to load new
    // dumps here. similar for tgeof, as the diagram below shows
    //
    //  rmax_geo      r=100.  r=100.                    r=rcam
    //
    //      |-----------|--------|----------------------> [cam]
    //      t0        tgeof    tgeoi                       t=0
    //
    double tgeoi = -1.e100;
    double tgeof = 0.;

    // TODO disallows large slow-light images
    int nsteps[nx][ny];

    // first pass through geodesics to find lengths
#pragma omp parallel for schedule(dynamic,2) collapse(2) reduction(max:tgeoi) reduction(max:maxsteplength) reduction(min:t0) reduction(min:tgeof) shared(nsteps)
    for (int i=0; i<nx; ++i) {
      for (int j=0; j<ny; ++j) {
        if (j==0) fprintf(stderr, "%d ", i);

        int nstep = 0;
        double dl;
        double X[NDIM], Xhalf[NDIM], Kcon[NDIM], Kconhalf[NDIM];
        init_XK(i,j, nx, ny, Xcam, params, fovx,fovy, X, Kcon);

        for (int k=0; k<NDIM; ++k) Kcon[k] *= freq;

        double tgeoitmp = 1.;
        double tgeoftmp = 1.;

        MULOOP Xhalf[mu] = X[mu];
        while (!stop_backward_integration(X, Xhalf, Kcon)) {
          dl = stepsize(X, Kcon);
          push_photon(X, Kcon, -dl, Xhalf, Kconhalf);
          nstep++;

          if (nstep > MAXNSTEP - 2) {
            fprintf(stderr, "MAXNSTEP exceeded on j=%d i=%d\n", j,i);
            break;
          }

          if (tgeoitmp > 0. && X[1] < log(100.)) tgeoitmp = X[0];
          if (tgeoftmp > 0. && X[1] > log(100.) && Kcon[1] < 0.) tgeoftmp = X[0]; // Kcon is for forward integration
        }

        nsteps[i][j] = --nstep;

        if (nstep > maxsteplength) maxsteplength = nstep;
        if (X[0] < t0) t0 = X[0];
        if (tgeoitmp < 0. && tgeoitmp > tgeoi) tgeoi = tgeoitmp;
        if (tgeoftmp < 0. && tgeoftmp < tgeof) tgeof = tgeoftmp;
        else if (tgeoftmp > 0. && X[0] < tgeof) tgeof = X[0];
      }
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "%g %g %g\n", t0, tgeoi, tgeof);

    // now create data structures for geodesics and light rays
    maxsteplength += 2;
    fprintf(stderr, "geodesic size = %g GB\n", 1. * sizeof(struct of_traj) * nx*ny * maxsteplength / 1024/1024/1024);
    struct of_traj *dtraj = malloc(sizeof(*dtraj) * nx*ny * maxsteplength);

    // TODO: if DTd is different, we'll have to calcluate this a different way
    int nconcurrentimgs = 2. + 1. * fabs(t0) / params.img_cadence;
    fprintf(stderr, "images size = %g GB\n", 1. * sizeof(struct of_image) * nx*ny * nconcurrentimgs / 1024/1024/1024);
    struct of_image *dimage = malloc(sizeof(*dimage) * nx*ny * nconcurrentimgs);

    // now populate the geodesic data structures
#pragma omp parallel for schedule(dynamic,2) collapse(2)
    for (int i=0; i<nx; ++i) {
      for (int j=0; j<ny; ++j) {
        if (j==0) fprintf(stderr, "%d ", i);

        int nstep = 0;
        double dl;
        double X[NDIM], Xhalf[NDIM], Kcon[NDIM], Kconhalf[NDIM];
        init_XK(i,j, nx, ny, Xcam, params, fovx,fovy, X, Kcon);
        for (int k=0; k<NDIM; ++k) Kcon[k] *= freq;

        MULOOP Xhalf[mu] = X[mu];
        while (!stop_backward_integration(X, Xhalf, Kcon)) {
          dl = stepsize(X, Kcon);
          push_photon(X, Kcon, -dl, Xhalf, Kconhalf);
          nstep++;

          int stepidx = imgindex(nstep,i,j,nx,ny);

          dtraj[stepidx].dl = dl * L_unit * HPL / (ME * CL * CL);
          for (int k=0; k<NDIM; ++k) {
            dtraj[stepidx].X[k] = X[k];
            dtraj[stepidx].Kcon[k] = Kcon[k];
            dtraj[stepidx].Xhalf[k] = Xhalf[k];
            dtraj[stepidx].Kconhalf[k] = Kconhalf[k];
          }

          if (nstep > MAXNSTEP - 2) {
            fprintf(stderr, "MAXNSTEP exceeded on j=%d i=%d\n", j,i);
            break;
          }
        }

      }
    }

    fprintf(stderr, "\n\nnow beginning radiative transfer calculation ...\n");

    // initialize state
    int nimg = 0, nopenimgs = 0;
    double last_img_target = tA - tgeof;
    double *target_times = malloc(sizeof(*target_times) * nconcurrentimgs);
    int *valid_images = malloc(sizeof(*valid_images) * nconcurrentimgs);
    for (int i=0; i<nconcurrentimgs; ++i) valid_images[i] = 0;

    fprintf(stderr, "first image will be produced for t = %g\n", last_img_target);

    // optionally start from restart file
    if( access("restart.h5", F_OK) != -1 ) {

      fprintf(stderr, "attempting to load restart file...\n");
      double ttA, ttB;
      read_restart("restart.h5", &ttA, &ttB, &last_img_target, &nopenimgs, &nimg,
            nconcurrentimgs, nconcurrentimgs*nx*ny,
            target_times, valid_images, dimage);
      update_data_until(&tA, &tB, ttA);
    }

    // now do radiative transfer
    while (1) {

      while ( last_img_target + t0 < tB ) {
        // add a new image if the final time for that image is before the final valid dump
        target_times[nimg] = last_img_target;
        if ( last_img_target + tgeoi < tf ) {
          valid_images[nimg] = 1;
          nopenimgs++;
          for (int i=0; i<nx; ++i) {
            for (int j=0; j<ny; ++j) {
              int pxidx = imgindex(nimg,i,j,nx,ny);
              dimage[pxidx].nstep = nsteps[i][j];
              dimage[pxidx].intensity = 0.;
              dimage[pxidx].tau = 0.;
              dimage[pxidx].tauF = 0.;
            }
          }
          if (++nimg == nconcurrentimgs) nimg = 0;
        }
        last_img_target += params.img_cadence;
      }

      for (int k=0; k<nconcurrentimgs; ++k) {  // images

        if (valid_images[k] == 0) continue;

        int do_output = 1;

#pragma omp parallel for schedule(dynamic,2) collapse(2)
        for (int i=0; i<nx; ++i) {
          for (int j=0; j<ny; ++j) {
            if (j==0) fprintf(stderr, "%d ", i);

            double ji,ki, jf,kf;
            double Xi[NDIM],Xhalf[NDIM],Xf[NDIM];
            double Kconi[NDIM],Kconhalf[NDIM],Kconf[NDIM];

            int pxidx = imgindex(k,i,j,nx,ny);

            while (dimage[pxidx].nstep > 1) {

              int stepidx = imgindex(dimage[pxidx].nstep,i,j,nx,ny);
              int pstepidx = imgindex(dimage[pxidx].nstep-1,i,j,nx,ny);

              for (int l=0; l<NDIM; ++l) {
                Xi[l]       = dtraj[stepidx].X[l];
                Xhalf[l]    = dtraj[stepidx].Xhalf[l];
                Xf[l]       = dtraj[pstepidx].X[l];
                Kconi[l]    = dtraj[stepidx].Kcon[l];
                Kconhalf[l] = dtraj[stepidx].Kconhalf[l];
                Kconf[l]    = dtraj[pstepidx].Kcon[l];
              }

              // adjust time. constant factor is to avoid floating point precision issues
              Xi[0] += target_times[k] + 1.e-5;
              Xhalf[0] += target_times[k] + 1.e-5;
              Xf[0] += target_times[k] + 1.e-5;

              // only integrate for points within the bounds
              if (Xi[0] < tA) {
                // this should only fire when we first start the program
                Xf[0] += tA - Xi[0];
                Xhalf[0] += tA - Xi[0];
                Xi[0] = tA;
              }
              if (Xi[0] >= tB) {
                if (Xf[0] >= tf) {
                  Xi[0] += tf - Xf[0];
                  Xhalf[0] += tf - Xf[0];
                  Xf[0] = tf;
                } else {
                  break;
                }
              }

              get_jkinv(Xi, Kconi, &ji, &ki);
              get_jkinv(Xf, Kconf, &jf, &kf);

              dimage[pxidx].intensity = 
                approximate_solve(dimage[pxidx].intensity, ji,ki,jf,kf, dtraj[stepidx].dl, &(dimage[pxidx].tau));
              
              // polarized transport
              if (! only_unpolarized) {
                evolve_N(Xi, Kconi, Xhalf, Kconhalf, Xf, Kconf, dtraj[stepidx].dl, dimage[pxidx].N_coord, &(dimage[pxidx].tauF));
                if (isnan(creal(dimage[pxidx].N_coord[0][0]))) {
                  exit(-2);
                }
              }

              dimage[pxidx].nstep -= 1;
            }

            if (dimage[pxidx].nstep != 1) {
              do_output = 0;
            }

          }
        }

        if (do_output) {

          // image, imageS, taus
          for (int i=0; i<nx; ++i) {
            for (int j=0; j<ny; ++j) {

              int pxidx = imgindex(k,i,j,nx,ny);
              int pstepidx = imgindex(dimage[pxidx].nstep,i,j,nx,ny);

              image[i*ny+j] = dimage[pxidx].intensity * pow(freqcgs, 3.);
              taus[i*ny+j] = dimage[pxidx].tau;

              if (! only_unpolarized) {
                double Stokes_I, Stokes_Q, Stokes_U, Stokes_V;
                project_N(dtraj[pstepidx].X, dtraj[pstepidx].Kcon, dimage[pxidx].N_coord, &Stokes_I, &Stokes_Q, &Stokes_U, &Stokes_V, params.rotcam);
                imageS[(i*ny+j)*NIMG+0] = Stokes_I * pow(freqcgs, 3.);
                imageS[(i*ny+j)*NIMG+1] = Stokes_Q * pow(freqcgs, 3.);
                imageS[(i*ny+j)*NIMG+2] = Stokes_U * pow(freqcgs, 3.);
                imageS[(i*ny+j)*NIMG+3] = Stokes_V * pow(freqcgs, 3.);
                imageS[(i*ny+j)*NIMG+4] = dimage[pxidx].tauF;
                if (params.qu_conv == 0) {
                  imageS[(i*ny+j)*NIMG+1] *= -1;
                  imageS[(i*ny+j)*NIMG+2] *= -1;
                }
              } else {
                imageS[(i*ny+j)*NIMG+0] = 0.;
                imageS[(i*ny+j)*NIMG+1] = 0.;
                imageS[(i*ny+j)*NIMG+2] = 0.;
                imageS[(i*ny+j)*NIMG+3] = 0.;
                imageS[(i*ny+j)*NIMG+4] = 0.;
              }
            }
          }

          //fprintf(stderr, "saving image %d at t = %g\n", k, target_times[k]);
          char dfname[256];
          snprintf(dfname, 255, params.outf, target_times[k]);
          dump(image, imageS, taus, dfname, scale, Xcam, fovx, fovy, &params);
          valid_images[k] = 0;
          nopenimgs--;
        }

      }

      if (nopenimgs < 1) break;
      update_data(&tA, &tB);

      // write a restart file however frequency as desired
      if (params.restart_int > 0. && tA > next_restart_after) {
        while (tA > next_restart_after) next_restart_after += params.restart_int;
        char rfname[256];
        snprintf(rfname, 200, "restarts/restart_%05.1f.h5", tA);
        write_restart(rfname, tA, tB, last_img_target, nopenimgs, nimg, 
              nconcurrentimgs, nconcurrentimgs*nx*ny,
              target_times, valid_images, dimage);
      }

    }

  // FAST LIGHT
  } else {

    // HALF-SIZE image for convergence comparison
    double *image_half;
    if (refine_level > 1) {
      nx = params.nx_min/2;
      ny = params.ny_min/2;
      image_half = calloc(nx*ny, sizeof(*image_half));

#pragma omp parallel for schedule(dynamic,1) collapse(2) shared(image,imageS)
      for (int i=0; i < nx; ++i) {
        for (int j=0; j < ny; ++j) {
          if (j==0) fprintf(stderr, "%d ", i);
          double Intensity = 0;
          double Is = 0, Qs = 0, Us = 0, Vs = 0;
          double Tau = 0, tauF = 0;

          get_pixel(i, j, nx, ny, Xcam, params,
                    fovx, fovy, freq, 1, scale,
                    &Intensity, &Is, &Qs, &Us, &Vs, &Tau, &tauF);

          image_half[i*ny+j] = Intensity;
        }
      }
    }

    // NORMAL IMAGE at n_min
    // TODO this is extraneous allocation in dynamic runs
    nx = params.nx_min;
    ny = params.ny_min;
    double *taus_min, *imageS_min, *image_min;
    if (refine_level > 1) {
      taus_min = calloc(nx*ny, sizeof(*taus_min));
      imageS_min = calloc(nx*ny*NIMG, sizeof(*imageS_min));
      image_min = calloc(nx*ny, sizeof(*image_min));
    } else {
      taus_min = taus;
      imageS_min = imageS;
      image_min = image;
    }
    double avg_val = 0;

#pragma omp parallel for schedule(dynamic,1) collapse(2) reduction(+:avg_val)
    for (int i=0; i < nx; ++i) {
      for (int j=0; j < ny; ++j) {
        if (j==0) fprintf(stderr, "%d ", i);

        double Intensity = 0;
        double Is = 0, Qs = 0, Us = 0, Vs = 0;
        double Tau = 0, tauF = 0;

        get_pixel(i, j, nx, ny, Xcam, params,
                  fovx, fovy, freq, only_unpolarized, scale,
                  &Intensity, &Is, &Qs, &Us, &Vs, &Tau, &tauF);

        if (refine_level == 1) {
          // At one refinement just save the pixel
          save_pixel(image, imageS, taus, i, j, nx, ny, 0,
                     Intensity, Is, Qs, Us, Vs, freqcgs, Tau, tauF);
        } else {
          // Otherwise keep the Stokes params for next refinement
          image_min[i*ny+j] = Intensity;
          taus_min[i*ny+j] = Tau;
          imageS_min[(i*ny+j)*NIMG+0] = Is;
          imageS_min[(i*ny+j)*NIMG+1] = Qs;
          imageS_min[(i*ny+j)*NIMG+2] = Us;
          imageS_min[(i*ny+j)*NIMG+3] = Vs;
          imageS_min[(i*ny+j)*NIMG+4] = tauF;

          avg_val += Intensity;
        }
      }
    }

    // NOTE: Only filled if using adaptive refinement
    avg_val /= nx*ny;

    // REFINE based on previous two images
    double *grandparent_image = image_half;

    double *parent_image = image_min;
    double *parent_taus = taus_min;
    double *parent_imageS = imageS_min;

    double *image_temp, *taus_temp, *imageS_temp;
    for (int refined_level = 1; refined_level < refine_level; refined_level++) {
      nx *= 2;
      ny *= 2;

      if (refined_level < refine_level - 1) {
        taus_temp = calloc(nx*ny, sizeof(*taus_temp));
        imageS_temp = calloc(nx*ny*NIMG, sizeof(*imageS_temp));
        image_temp = calloc(nx*ny, sizeof(*image_temp));
      }

#pragma omp parallel for schedule(dynamic,1) collapse(2)
      for (int i=0; i < nx; ++i) {
        for (int j=0; j < ny; ++j) {
          if (j==0) fprintf(stderr, "%d ", i);

          double Intensity = 0;
          double Is = 0, Qs = 0, Us = 0, Vs = 0;
          double Tau = 0, tauF = 0;

          int ig = i/4, jg = j/4, nyg = ny/4;
          double j_grandparent = grandparent_image[ig*nyg+jg];
          // Anchor parent quads on grandparent pixels i.e. even 4th pixels of current image
          //int ip = (i/4)*2, jp = (j/4)*2, nyp=ny/2;
          // Anchor parent quads continuously if only looking for differences
          int ip = i/2, jp = j/2, nyp = ny/2;
          double j_parent = (parent_image[(ip)*nyp+(jp)] + parent_image[(ip+1)*nyp+(jp)]
                            + parent_image[(ip)*nyp+(jp+1)] + parent_image[(ip+1)*nyp+(jp+1)])/4;
          double dev_parent = 0;
          if (fabs(parent_image[(ip)*nyp+(jp)] - j_parent) > dev_parent) dev_parent = fabs(parent_image[(ip)*nyp+(jp)] - j_parent);
          if (fabs(parent_image[(ip+1)*nyp+(jp)] - j_parent) > dev_parent) dev_parent = fabs(parent_image[(ip+1)*nyp+(jp)] - j_parent);
          if (fabs(parent_image[(ip)*nyp+(jp+1)] - j_parent) > dev_parent) dev_parent = fabs(parent_image[(ip)*nyp+(jp+1)] - j_parent);
          if (fabs(parent_image[(ip+1)*nyp+(jp+1)] - j_parent) > dev_parent) dev_parent = fabs(parent_image[(ip+1)*nyp+(jp+1)] - j_parent);

          //fprintf(stderr, "Grandparent px: %g, Parent av: %g dev: %g, prop %g\n", j_grandparent, j_parent, dev_parent, dev_parent/j_parent);

          // 2 refinement criteria, must meet both to bother refining
          // 1. Difference grandparent -> parent refinement must be > refine_rel
          // 2. Parent flux is less than the average value * refine_cut
          // fabs(j_grandparent - j_parent) / j_parent > params.refine_rel
          if (dev_parent / j_parent > params.refine_rel &&
              fabs(j_parent) / avg_val > params.refine_cut) {
            get_pixel(i, j, nx, ny, Xcam, params,
                      fovx, fovy, freq, only_unpolarized, scale,
                      &Intensity, &Is, &Qs, &Us, &Vs, &Tau, &tauF);
          } else {
            // TODO like, bilinear at least
            Intensity = parent_image[(i/2)*(ny/2)+j/2];
            //fprintf(stderr, "Skipping Pixel %d %d, assigning %g\n", i, j, Intensity);
            Tau = parent_taus[(i/2)*(ny/2)+j/2];
            Is = parent_imageS[((i/2)*(ny/2)+j/2)*NIMG+0];
            Qs = parent_imageS[((i/2)*(ny/2)+j/2)*NIMG+1];
            Us = parent_imageS[((i/2)*(ny/2)+j/2)*NIMG+2];
            Vs = parent_imageS[((i/2)*(ny/2)+j/2)*NIMG+3];
            tauF = parent_imageS[((i/2)*(ny/2)+j/2)*NIMG+4];
          }

          if (refined_level == refine_level - 1) {
            save_pixel(image, imageS, taus, i, j, nx, ny, 0,
                       Intensity, Is, Qs, Us, Vs, freqcgs, Tau, tauF);
          } else {
            image_temp[i*ny+j] = Intensity;
            taus_temp[i*ny+j] = Tau;
            imageS_temp[(i*ny+j)*NIMG+0] = Is;
            imageS_temp[(i*ny+j)*NIMG+1] = Qs;
            imageS_temp[(i*ny+j)*NIMG+2] = Us;
            imageS_temp[(i*ny+j)*NIMG+3] = Vs;
            imageS_temp[(i*ny+j)*NIMG+4] = tauF;
          }
        }
      }

      if (refined_level < refine_level - 1) {
        // CLEAN UP YO TOYS
        free(grandparent_image); free(parent_taus); free(parent_imageS);
        grandparent_image = parent_image;

        parent_image = image_temp;
        parent_taus = taus_temp;
        parent_imageS = imageS_temp;
      }
    }

    print_image_stats(image, imageS, nx, ny, params, scale);

    // don't dump if we've been asked to quench output. useful for batch jobs
    // like when fitting light curve fluxes
    if (!quench_output) {
      // dump result. if specified, also output ppm image
      dump(image, imageS, taus, params.outf, scale, Xcam, fovx, fovy, &params);
      if (params.add_ppm) {
        // TODO respect filename from params?
        make_ppm(image, freq, nx, ny, "ipole_lfnu.ppm");
      }
    }
  } // SLOW_LIGHT

  time = omp_get_wtime() - time;
  printf("Total wallclock time: %g s\n\n", time);

  return 0;
}

void get_pixel(int i, int j, int nx, int ny, double Xcam[NDIM], Params params,
               double fovx, double fovy, double freq, int only_unpolarized, double scale,
               double *Intensity, double *Is, double *Qs, double *Us, double *Vs,
               double *Tau, double *tauF)
{
  double X[NDIM], Kcon[NDIM];
  double complex N_coord[NDIM][NDIM];

  // Integrate backward to find geodesic trajectory
  init_XK(i,j, nx, ny, Xcam, params, fovx,fovy, X, Kcon);
  struct of_traj *traj = calloc(MAXNSTEP, sizeof(struct of_traj));
  MULOOP Kcon[mu] *= freq;
  int nstep = trace_geodesic(X, Kcon, traj);

  // Integrate emission forward along trajectory
  integrate_emission(traj, nstep, only_unpolarized, Intensity, Tau, tauF, N_coord);

  // Record values along the geodesic if requested
  // Figure out how to do this with adaptive...
  if (params.trace) {
    int stride = params.trace_stride;
    if (params.trace_i < 0 || params.trace_j < 0) { // If no single point is specified
      if (i % stride == 0 && j % stride == 0) { // Save every stride pixels
#pragma omp critical
        dump_var_along(i/stride, j/stride, nstep, traj, nx/stride, ny/stride, scale, Xcam, fovx, fovy, &params);
      }
    } else {
      if (i == params.trace_i && j == params.trace_j) { // Save just the one
#pragma omp critical
        dump_var_along(0, 0, nstep, traj, 1, 1, scale, Xcam, fovx, fovy, &params);
      }
    }
  }

  if (!only_unpolarized) {
    project_N(traj[0].X, traj[0].Kcon, N_coord, Is, Qs, Us, Vs, params.rotcam);
  }

  free(traj);
}

void save_pixel(double *image, double *imageS, double *taus, int i, int j, int nx, int ny, int only_intensity,
                double Intensity, double Is, double Qs, double Us, double Vs,
                double freqcgs, double Tau, double tauF)
{
  // deposit the intensity and Stokes parameter in pixel
  image[i*ny+j] = Intensity * pow(freqcgs, 3);

  if (!only_intensity) {
    taus[i*ny+j] = Tau;

    imageS[(i*ny+j)*NIMG+0] = Is * pow(freqcgs, 3);
    if (params.qu_conv == 0) {
      imageS[(i*ny+j)*NIMG+1] = -Qs * pow(freqcgs, 3);
      imageS[(i*ny+j)*NIMG+2] = -Us * pow(freqcgs, 3);
    } else {
      imageS[(i*ny+j)*NIMG+1] = Qs * pow(freqcgs, 3);
      imageS[(i*ny+j)*NIMG+2] = Us * pow(freqcgs, 3);
    }
    imageS[(i*ny+j)*NIMG+3] = Vs * pow(freqcgs, 3);
    imageS[(i*ny+j)*NIMG+4] = tauF;

    if (isnan(imageS[(i*ny+j)*NIMG+0])) {
      fprintf(stderr, "NaN in image! Exiting.\n");
      exit(-1);
    }
  }
}

void print_image_stats(double *image, double *imageS, int nx, int ny, Params params, double scale)
{
  double Ftot = 0.;
  double Ftot_unpol = 0.;
  double Imax = 0.0;
  double Iavg = 0.0;
  double Qtot = 0.;
  double Utot = 0.;
  double Vtot = 0.;
  int imax = 0;
  int jmax = 0;
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      Ftot_unpol += image[i*ny+j]*scale;
      Ftot += imageS[(i*ny+j)*NIMG+0] * scale;
      Iavg += imageS[(i*ny+j)*NIMG+0];
      Qtot += imageS[(i*ny+j)*NIMG+1] * scale;
      Utot += imageS[(i*ny+j)*NIMG+2] * scale;
      Vtot += imageS[(i*ny+j)*NIMG+3] * scale;
      if (imageS[(i*ny+j)*NIMG+0] > Imax) {
        imax = i;
        jmax = j;
        Imax = imageS[(i*ny+j)*NIMG+0];
      }
    }
  }

  // output normal flux quantities
  fprintf(stderr, "\nscale = %e\n", scale);
  fprintf(stderr, "imax=%d jmax=%d Imax=%g Iavg=%g\n", imax, jmax, Imax, Iavg/(params.nx*params.ny));
  fprintf(stderr, "freq: %g Ftot: %g Jy (%g Jy unpol xfer) scale=%g\n", params.freqcgs, Ftot, Ftot_unpol, scale);
  fprintf(stderr, "nuLnu = %g erg/s\n", 4.*M_PI*Ftot * params.dsource * params.dsource * JY * params.freqcgs);

  // output polarized transport information
  double LPfrac = 100.*sqrt(Qtot*Qtot+Utot*Utot)/Ftot;
  double CPfrac = 100.*Vtot/Ftot;
  fprintf(stderr, "I,Q,U,V [Jy]: %g %g %g %g\n", Ftot, Qtot, Utot, Vtot);
  fprintf(stderr, "LP,CP [%%]: %g %g\n", LPfrac, CPfrac);
}
