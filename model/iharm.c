#include "decs.h"
#include "hdf5_utils.h"

#define NVAR (10)
#define USE_FIXED_TPTE (0)
#define USE_MIXED_TPTE (1)

// these will be overwritten by anything found in par.c (or in runtime parameter file)
static double tp_over_te = 3.; 
static double trat_small = 1.;
static double trat_large = 30.;

// ELECTRONS -> 
//    0 : constant TP_OVER_TE
//    1 : use dump file model (kawazura?)
//    2 : use mixed TP_OVER_TE (beta model)
static int RADIATION, ELECTRONS;

static inline __attribute__((always_inline)) void set_dxdX_metric(double X[NDIM], double dxdX[NDIM][NDIM], int metric);
static inline __attribute__((always_inline)) void gcov_ks(double r, double th, double gcov[NDIM][NDIM]);
double gdet_zone(int i, int j, int k);
void ijktoX(int i, int j, int k, double X[NDIM]);
void Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM]) ;
int X_in_domain(double X[NDIM]);
void interp_fourv(double X[NDIM], double ****fourv, double Fourv[NDIM]) ;
double interp_scalar(double X[NDIM], double ***var) ;
static double game, gamp;

// metric parameters 
//  note: if METRIC_eKS, then the code will use "expoential KS" coordinates
//        defined by x^a = { x^0, log(x^1), x^2, x^3 } where x^0, x^1, x^2,
//        x^3 are normal KS coordinates. in addition, you must set METRIC_* 
//        as well in order to specify how Xtoijk should work.
int METRIC_eKS;
static int DEREFINE_POLES, METRIC_MKS3;
static double poly_norm, poly_xt, poly_alpha, mks_smooth; // mmks
static double mks3R0, mks3H0, mks3MY1, mks3MY2, mks3MP0; // mks3

static int dumpmin, dumpmax, dumpidx, dumpskip;

static hdf5_blob fluid_header = { 0 };
static double MBH, Mdotedd, tp_over_te, Thetae_unit;
#define NSUP (3)

struct of_data {
  double t;
  double ****bcon;
  double ****bcov;
  double ****ucon;
  double ****ucov;
  double ****p;
  double ***ne;
  double ***thetae;
  double ***b;
};
static int nloaded = 0;

struct of_data dataA, dataB, dataC;
struct of_data *data[NSUP];
  
void load_iharm_data(int n, char *, int dumpidx, int verbose);

void parse_input(int argc, char *argv[], Params *params)
{

  // if params has been loaded, just read from it
  if ( params->loaded ) {
    thetacam = params->thetacam;
    freqcgs = params->freqcgs;
    MBH = params->MBH * MSUN;
    M_unit = params->M_unit;
    strcpy(fnam, params->dump);
    tp_over_te = params->tp_over_te;
    trat_small = params->trat_small;
    trat_large = params->trat_large;
    counterjet = params->counterjet;

    dumpmin = params->dump_min;
    dumpmax = params->dump_max;
    dumpskip = params->dump_skip;
    dumpidx = dumpmin;

    return;
  }

  fprintf(stderr, "argc: %d\n", argc);

  // 7 -> defaults, 8 -> fixed, 9 -> mixed
  if (argc < 7 || argc > 9) {
    fprintf(stderr, "ERROR format is\n");
    fprintf(stderr, "  ipole theta[deg] freq[cgs] MBH[Msolar] M_unit[g] filename counterjet [...thermodynamics]\n");
    exit(-1);
  }

  sscanf(argv[1], "%lf", &thetacam);
  sscanf(argv[2], "%lf", &freqcgs);
  sscanf(argv[3], "%lf", &MBH);
  MBH *= MSUN;
  sscanf(argv[4], "%lf", &M_unit);
  strcpy(fnam, argv[5]);
  sscanf(argv[6], "%d",  &counterjet);

  if (argc == 8) {
    sscanf(argv[7], "%lf", &tp_over_te);
  }

  if (argc == 9) {
    sscanf(argv[7], "%lf", &trat_small);
    sscanf(argv[8], "%lf", &trat_large);
  }
}

double set_tinterp_ns(double X[NDIM], int *nA, int *nB)
{
  #if SLOW_LIGHT
  if (X[0] < data[1]->t) {
    *nA = 0; *nB = 1;
  } else {
    *nA = 1; *nB = 2;
  }
  double tinterp = ( X[0] - data[*nA]->t ) / ( data[*nB]->t - data[*nA]->t );
  if (tinterp < 0.) tinterp = 0.; //  in slow light, when we reset based on tB, sometimes we overshoot
  if (tinterp > 1.) tinterp = 1.; //  TODO, this should really only happen at r >> risco, but still...
  return tinterp;
  #else
  *nA = 0;
  *nB = 0;
  return 0.;
  #endif // SLOW_LIGHT
}

void update_data(double *tA, double *tB)
{
  #if SLOW_LIGHT
  // Reorder dataA, dataB, dataC in data[]
  if (nloaded % 3 == 0) {
    data[0] = &dataB;
    data[1] = &dataC;
    data[2] = &dataA;
  } else if (nloaded % 3 == 1) {
    data[0] = &dataC;
    data[1] = &dataA;
    data[2] = &dataB;
  } else {
    data[0] = &dataA;
    data[1] = &dataB;
    data[2] = &dataC;
  }
  int nextdumpidx = dumpidx;
  dumpidx += dumpskip;
  if (nextdumpidx > dumpmax) {
    load_iharm_data(2, fnam, --nextdumpidx, 0);
    data[2]->t += 1.;
  } else {
    load_iharm_data(2, fnam, nextdumpidx, 0);
  }
  *tA = data[0]->t;
  *tB = data[1]->t;
  fprintf(stderr, "loaded data (dump %d) (%g < t < %g)\n", nextdumpidx, *tA, *tB);
  #else // FAST LIGHT
  if (nloaded % 3 == 0) {
    data[0] = &dataA;
    data[1] = &dataB;
    data[2] = &dataC;
  } else if (nloaded % 3 == 1) {
    data[0] = &dataB;
    data[1] = &dataC;
    data[2] = &dataA;
  } else if (nloaded % 3 == 2) {
    data[0] = &dataC;
    data[1] = &dataA;
    data[2] = &dataB;
  } else {
    printf("Fail! nloaded = %i nloaded mod 3 = %i\n", nloaded, nloaded % 3);
  }
  data[2]->t = data[1]->t + DTd;
  #endif 
}

void set_dxdX(double X[NDIM], double dxdX[NDIM][NDIM])
{
  set_dxdX_metric(X, dxdX, 0);
}

static inline __attribute__((always_inline)) void set_dxdX_metric(double X[NDIM], double dxdX[NDIM][NDIM], int metric)
{
  // Jacobian with respect to KS basis where X is given in
  // non-KS basis
  MUNULOOP dxdX[mu][nu] = 0.;

  if ( METRIC_eKS && metric==0 ) {

    MUNULOOP dxdX[mu][nu] = mu==nu ? 1 : 0;
    dxdX[1][1] = exp(X[1]);
    dxdX[2][2] = M_PI;

  } else if ( METRIC_MKS3 ) {
   
    // mks3 ..
    dxdX[0][0] = 1.;
    dxdX[1][1] = exp(X[1]);
    dxdX[2][1] = -(pow(2.,-1. + mks3MP0)*exp(X[1])*mks3H0*mks3MP0*(mks3MY1 - 
              mks3MY2)*pow(M_PI,2)*pow(exp(X[1]) + mks3R0,-1 - mks3MP0)*(-1 + 
              2*X[2])*1./tan((mks3H0*M_PI)/2.)*pow(1./cos(mks3H0*M_PI*(-0.5 + (mks3MY1 + 
              (pow(2,mks3MP0)*(-mks3MY1 + mks3MY2))/pow(exp(X[1]) + mks3R0,mks3MP0))*(1 - 
              2*X[2]) + X[2])),2));
    dxdX[2][2]= (mks3H0*pow(M_PI,2)*(1 - 2*(mks3MY1 + (pow(2,mks3MP0)*(-mks3MY1 +
             mks3MY2))/pow(exp(X[1]) + mks3R0,mks3MP0)))*1./tan((mks3H0*M_PI)/2.)*
             pow(1./cos(mks3H0*M_PI*(-0.5 + (mks3MY1 + (pow(2,mks3MP0)*(-mks3MY1 +
             mks3MY2))/pow(exp(X[1]) + mks3R0,mks3MP0))*(1 - 2*X[2]) + X[2])),2))/2.;
    dxdX[3][3] = 1.;

  } else if (DEREFINE_POLES) {

    // mmks
    dxdX[0][0] = 1.;
    dxdX[1][1] = exp(X[1]);
    dxdX[2][1] = -exp(mks_smooth*(startx[1]-X[1]))*mks_smooth*(
      M_PI/2. -
      M_PI*X[2] +
      poly_norm*(2.*X[2]-1.)*(1+(pow((-1.+2*X[2])/poly_xt,poly_alpha))/(1 + poly_alpha)) -
      1./2.*(1. - hslope)*sin(2.*M_PI*X[2])
      );
    dxdX[2][2] = M_PI + (1. - hslope)*M_PI*cos(2.*M_PI*X[2]) +
      exp(mks_smooth*(startx[1]-X[1]))*(
        -M_PI +
        2.*poly_norm*(1. + pow((2.*X[2]-1.)/poly_xt,poly_alpha)/(poly_alpha+1.)) +
        (2.*poly_alpha*poly_norm*(2.*X[2]-1.)*pow((2.*X[2]-1.)/poly_xt,poly_alpha-1.))/((1.+poly_alpha)*poly_xt) -
        (1.-hslope)*M_PI*cos(2.*M_PI*X[2])
        );
    dxdX[3][3] = 1.;

  } else {

    // mks
    dxdX[0][0] = 1.;
    dxdX[1][1] = exp(X[1]);
    dxdX[2][2] = M_PI - (hslope - 1.)*M_PI*cos(2.*M_PI*X[2]);
    dxdX[3][3] = 1.;

  }

}

void gcov_func(double X[NDIM], double gcov[NDIM][NDIM])
{
  // returns g_{munu} at location specified by X
 
  MUNULOOP gcov[mu][nu] = 0.;
    
  double r, th;

  // despite the name, get equivalent values for
  // r, th for kerr coordinate system
  bl_coord(X, &r, &th);

  // compute ks metric
  gcov_ks(r, th, gcov);

  // convert from ks metric to mks/mmks
  double dxdX[NDIM][NDIM];
  set_dxdX(X, dxdX);

  double gcov_ks[NDIM][NDIM];
  MUNULOOP {
    gcov_ks[mu][nu] = gcov[mu][nu];
    gcov[mu][nu] = 0.;
  }

  MUNULOOP {
    for (int lam=0; lam<NDIM; ++lam) {
      for (int kap=0; kap<NDIM; ++kap) {
        gcov[mu][nu] += gcov_ks[lam][kap]*dxdX[lam][mu]*dxdX[kap][nu];
      }
    }
  }
}

// return the gdet associated with zone coordinates for the zone at
// i,j,k
double gdet_zone(int i, int j, int k)
{
  // get the X for the zone (in geodesic coordinates for bl_coord) 
  // and in zone coordinates (for set_dxdX_metric)
  double X[NDIM], Xzone[NDIM];
  ijktoX(i,j,k, X);
  Xzone[0] = 0.;
  Xzone[1] = startx[1] + (i+0.5)*dx[1];
  Xzone[2] = startx[2] + (j+0.5)*dx[2];
  Xzone[3] = startx[3] + (k+0.5)*dx[3];

  // then get gcov for the zone (in zone coordinates)
  double gcovKS[NDIM][NDIM], gcov[NDIM][NDIM];
  double r, th;
  double dxdX[NDIM][NDIM];
  MUNULOOP gcovKS[mu][nu] = 0.;
  MUNULOOP gcov[mu][nu] = 0.;
  bl_coord(X, &r, &th);
  gcov_ks(r, th, gcovKS);
  set_dxdX_metric(Xzone, dxdX, 1);
  MUNULOOP {
    for (int lam=0; lam<NDIM; ++lam) {
      for (int kap=0; kap<NDIM; ++kap) {
        gcov[mu][nu] += gcovKS[lam][kap]*dxdX[lam][mu]*dxdX[kap][nu];
      }
    }
  }

  return gdet_func(gcov); 
}

// compute KS metric at point (r,th) in KS coordinates (cyclic in t, ph)
static inline __attribute__((always_inline)) void gcov_ks(double r, double th, double gcov[NDIM][NDIM])
{
  double cth = cos(th);
  double sth = sin(th);

  double s2 = sth*sth;
  double rho2 = r*r + a*a*cth*cth;

  // compute ks metric for ks coordinates (cyclic in t,phi)
  gcov[0][0] = -1. + 2.*r/rho2;
  gcov[0][1] = 2.*r/rho2;
  gcov[0][3] = -2.*a*r*s2/rho2;

  gcov[1][0] = gcov[0][1];
  gcov[1][1] = 1. + 2.*r/rho2;
  gcov[1][3] = -a*s2*(1. + 2.*r/rho2);

  gcov[2][2] = rho2;

  gcov[3][0] = gcov[0][3];
  gcov[3][1] = gcov[1][3];
  gcov[3][3] = s2*(rho2 + a*a*s2*(1. + 2.*r/rho2));
}


void get_connection(double X[4], double lconn[4][4][4])
{
  get_connection_num(X, lconn);
}

double get_dump_t(char *fnam, int dumpidx)
{
  char fname[256];
  snprintf(fname, 255, fnam, dumpidx);
  double t = -1.;

  if ( hdf5_open(fname) < 0 ) {
    fprintf(stderr, "! unable to open file %s. Exiting!\n", fname);
    exit(-1);
  }

  hdf5_set_directory("/");
  hdf5_read_single_val(&t, "/t", H5T_IEEE_F64LE);
  hdf5_close();

  return t;
}

void init_model(double *tA, double *tB)
{
  void init_iharm_grid(char *, int);

  // set up initial ordering of data[]
  data[0] = &dataA;
  data[1] = &dataB;
  data[2] = &dataC;

  // set up grid for fluid data
  fprintf(stderr, "reading data header...\n");
  init_iharm_grid(fnam, dumpmin);
  fprintf(stderr, "success\n");

  // set all dimensional quantities from loaded parameters
  set_units();

  // read fluid data
  fprintf(stderr, "reading data...\n");
  load_iharm_data(0, fnam, dumpidx, 1);
  dumpidx += dumpskip;
  #if SLOW_LIGHT
  update_data(tA, tB);
  update_data(tA, tB);
  tf = get_dump_t(fnam, dumpmax) - 1.e-5;
  #else // FAST LIGHT
  data[2]->t =10000.;
  #endif // SLOW_LIGHT
  fprintf(stderr, "success\n");

  // horizon radius
  Rh = 1 + sqrt(1. - a * a);
}

/*
 
  these supply basic model data to ipole

*/
void get_model_fourv(double X[NDIM], double Ucon[NDIM], double Ucov[NDIM],
                                     double Bcon[NDIM], double Bcov[NDIM])
{
  double gcov[NDIM][NDIM], gcon[NDIM][NDIM];

  gcov_func(X, gcov);
  gcon_func(gcov, gcon);

  // If we're outside of the logical domain, default to
  // normal observer velocity for Ucon/Ucov and default
  // Bcon/Bcov to zero.
  if ( X_in_domain(X) == 0 ) {

    Ucov[0] = -1./sqrt(-gcov[0][0]);
    Ucov[1] = 0.;
    Ucov[2] = 0.;
    Ucov[3] = 0.;

    for (int mu=0; mu<NDIM; ++mu) {
      Ucon[0] = Ucov[mu] * gcon[0][mu];
      Ucon[1] = Ucov[mu] * gcon[1][mu];
      Ucon[2] = Ucov[mu] * gcon[2][mu];
      Ucon[3] = Ucov[mu] * gcon[3][mu];
      Bcon[mu] = 0.;
      Bcov[mu] = 0.;
    }
   
    return;
  }

  // Set Ucon and get Ucov by lowering

  // interpolate primitive variables first
  double U1A, U2A, U3A, U1B, U2B, U3B, tfac;
  double Vcon[NDIM];
  int nA, nB;
  tfac = set_tinterp_ns(X, &nA, &nB);
  U1A = interp_scalar(X, data[nA]->p[U1]);
  U2A = interp_scalar(X, data[nA]->p[U2]);
  U3A = interp_scalar(X, data[nA]->p[U3]);
  U1B = interp_scalar(X, data[nB]->p[U1]);
  U2B = interp_scalar(X, data[nB]->p[U2]);
  U3B = interp_scalar(X, data[nB]->p[U3]);
  Vcon[1] = tfac*U1A + (1. - tfac)*U1B;
  Vcon[2] = tfac*U2A + (1. - tfac)*U2B;
  Vcon[3] = tfac*U3A + (1. - tfac)*U3B;

  // translate to four velocity
  double VdotV = 0.;
  for (int i = 1; i < NDIM; i++)
    for (int j = 1; j < NDIM; j++)
      VdotV += gcov[i][j] * Vcon[i] * Vcon[j];
  double Vfac = sqrt(-1. / gcon[0][0] * (1. + fabs(VdotV)));
  Ucon[0] = -Vfac * gcon[0][0];
  for (int i = 1; i < NDIM; i++)
    Ucon[i] = Vcon[i] - Vfac * gcon[0][i];

  // lower
  lower(Ucon, gcov, Ucov);

  // Now set Bcon and get Bcov by lowering

  // interpolate primitive variables first
  double B1A, B2A, B3A, B1B, B2B, B3B, Bcon1, Bcon2, Bcon3;
  tfac = set_tinterp_ns(X, &nA, &nB);
  B1A = interp_scalar(X, data[nA]->p[B1]);
  B2A = interp_scalar(X, data[nA]->p[B2]);
  B3A = interp_scalar(X, data[nA]->p[B3]);
  B1B = interp_scalar(X, data[nB]->p[B1]);
  B2B = interp_scalar(X, data[nB]->p[B2]);
  B3B = interp_scalar(X, data[nB]->p[B3]);
  Bcon1 = tfac*B1A + (1. - tfac)*B1B;
  Bcon2 = tfac*B2A + (1. - tfac)*B2B;
  Bcon3 = tfac*B3A + (1. - tfac)*B3B;

  // get Bcon
  Bcon[0] = Bcon1*Ucov[1] + Bcon2*Ucov[2] + Bcon3*Ucov[3];
  Bcon[1] = (Bcon1 + Ucon[1] * Bcon[0]) / Ucon[0];
  Bcon[2] = (Bcon2 + Ucon[2] * Bcon[0]) / Ucon[0];
  Bcon[3] = (Bcon3 + Ucon[3] * Bcon[0]) / Ucon[0];

  // lower
  lower(Bcon, gcov, Bcov);
}

double get_model_thetae(double X[NDIM])
{
  if ( X_in_domain(X) == 0 ) return 0.;
  
  double thetaeA, thetaeB, tfac;
  int nA, nB;
  tfac = set_tinterp_ns(X, &nA, &nB);
  thetaeA = interp_scalar(X, data[nA]->thetae);
  thetaeB = interp_scalar(X, data[nB]->thetae);

  double thetae = tfac*thetaeA + (1. - tfac)*thetaeB;
  if (thetae < 0.) {
    printf("thetae negative!\n");
    printf("X[] = %g %g %g %g\n", X[0], X[1], X[2], X[3]);
    printf("t = %e %e %e\n", data[0]->t, data[1]->t, data[2]->t);
    printf("thetae = %e tfac = %e thetaeA = %e thetaeB = %e nA = %i nB = %i\n",
    thetae, tfac, thetaeA, thetaeB, nA, nB);
  }

  if (thetaeA < 0 || thetaeB < 0) fprintf(stderr, "TETE %g %g\n", thetaeA, thetaeB);

  return tfac*thetaeA + (1. - tfac)*thetaeB;
}

//b field strength in Gauss
double get_model_b(double X[NDIM])
{
  if ( X_in_domain(X) == 0 ) return 0.;
  
  double bA, bB, tfac;
  int nA, nB;
  tfac = set_tinterp_ns(X, &nA, &nB);
  bA = interp_scalar(X, data[nA]->b);
  bB = interp_scalar(X, data[nB]->b);

  return tfac*bA + (1. - tfac)*bB;
}

double get_model_ne(double X[NDIM])
{
  if ( X_in_domain(X) == 0 ) return 0.;

  double neA, neB, tfac;
  int nA, nB;
  tfac = set_tinterp_ns(X, &nA, &nB);
  neA = interp_scalar(X, data[nA]->ne);
  neB = interp_scalar(X, data[nB]->ne);
  return tfac*neA + (1. - tfac)*neB;
}


/** HARM utilities **/


/********************************************************************

        Interpolation routines

 ********************************************************************/

/* return fluid four-vector in simulation units */
void interp_fourv(double X[NDIM], double ****fourv, double Fourv[NDIM]){
  double del[NDIM],b1,b2,b3,d1,d2,d3,d4;
  int i, j, k, ip1, jp1, kp1;

  /* find the current zone location and offsets del[0], del[1] */
  Xtoijk(X, &i, &j, &k, del);

  // since we read from data, adjust i,j,k for ghost zones
  i += 1;
  j += 1;
  k += 1;

  ip1 = i + 1;
  jp1 = j + 1;
  kp1 = k + 1;
  
  b1 = 1.-del[1];
  b2 = 1.-del[2];
  b3 = 1.-del[3];

  d1 = b1*b2;
  d3 = del[1] * b2;
  d2 = del[2] * b1;
  d4 = del[1] * del[2];


  /* Interpolate along x1,x2 first */
  Fourv[0] = d1*fourv[i][j][k][0] + d2*fourv[i][jp1][k][0] + d3*fourv[ip1][j][k][0] + d4*fourv[ip1][jp1][k][0];
  Fourv[1] = d1*fourv[i][j][k][1] + d2*fourv[i][jp1][k][1] + d3*fourv[ip1][j][k][1] + d4*fourv[ip1][jp1][k][1];
  Fourv[2] = d1*fourv[i][j][k][2] + d2*fourv[i][jp1][k][2] + d3*fourv[ip1][j][k][2] + d4*fourv[ip1][jp1][k][2];
  Fourv[3] = d1*fourv[i][j][k][3] + d2*fourv[i][jp1][k][3] + d3*fourv[ip1][j][k][3] + d4*fourv[ip1][jp1][k][3];

  /* Now interpolate above in x3 */
  Fourv[0] = b3*Fourv[0] + del[3]*(d1*fourv[i][j][kp1][0] + d2*fourv[i][jp1][kp1][0] + d3*fourv[ip1][j][kp1][0] + d4*fourv[ip1][jp1][kp1][0]);
  Fourv[1] = b3*Fourv[1] + del[3]*(d1*fourv[i][j][kp1][1] + d2*fourv[i][jp1][kp1][1] + d3*fourv[ip1][j][kp1][1] + d4*fourv[ip1][jp1][kp1][1]);
  Fourv[2] = b3*Fourv[2] + del[3]*(d1*fourv[i][j][kp1][2] + d2*fourv[i][jp1][kp1][2] + d3*fourv[ip1][j][kp1][2] + d4*fourv[ip1][jp1][kp1][2]);
  Fourv[3] = b3*Fourv[3] + del[3]*(d1*fourv[i][j][kp1][3] + d2*fourv[i][jp1][kp1][3] + d3*fourv[ip1][j][kp1][3] + d4*fourv[ip1][jp1][kp1][3]);
  //new

  //no interpolation of vectors at all
 
  //Fourv[0]=fourv[i][j][k][0];
  //Fourv[1]=fourv[i][j][k][1];
  //Fourv[2]=fourv[i][j][k][2];
  //Fourv[3]=fourv[i][j][k][3];
  
}

/* return scalar in cgs units */
double interp_scalar(double X[NDIM], double ***var)
{
  double del[NDIM],b1,b2,interp;
  int i, j, k, ip1, jp1, kp1;

  // zone and offset from X
  Xtoijk(X, &i, &j, &k, del);

  // since we read from data, adjust i,j,k for ghost zones
  i += 1;
  j += 1;
  k += 1;

  ip1 = i+1;
  jp1 = j+1;
  kp1 = k+1;

  b1 = 1.-del[1];
  b2 = 1.-del[2];

  // interpolate in x1 and x2
  interp = var[i][j][k]*b1*b2 + 
    var[i][jp1][k]*b1*del[2] + 
    var[ip1][j][k]*del[1]*b2 + 
    var[ip1][jp1][k]*del[1]*del[2];

  // then interpolate in x3
  interp = (1.-del[3])*interp + 
        del[3]*(var[i  ][j  ][kp1]*b1*b2 +
      var[i  ][jp1][kp1]*del[2]*b1 +
      var[ip1][j  ][kp1]*del[1]*b2 +
      var[ip1][jp1][kp1]*del[1]*del[2]);
  
  return interp;
}

/***********************************************************************************

          End interpolation routines

 ***********************************************************************************/

int X_in_domain(double X[NDIM]) {
  // returns 1 if X is within the computational grid.
  // checks different sets of coordinates depending on
  // specified grid coordinates

  if (METRIC_eKS) {
    double XG[4] = { 0 };
    double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };

    if (METRIC_MKS3) {
      // if METRIC_MKS3, ignore theta boundaries
      double H0 = mks3H0, MY1 = mks3MY1, MY2 = mks3MY2, MP0 = mks3MP0;
      double KSx1 = Xks[1], KSx2 = Xks[2];
      XG[0] = Xks[0];
      XG[1] = log(Xks[1] - mks3R0);
      XG[2] = (-(H0*pow(KSx1,MP0)*M_PI) - pow(2,1 + MP0)*H0*MY1*M_PI +
        2*H0*pow(KSx1,MP0)*MY1*M_PI + pow(2,1 + MP0)*H0*MY2*M_PI +
        2*pow(KSx1,MP0)*atan(((-2*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI))/(2.*
        H0*(-pow(KSx1,MP0) - pow(2,1 + MP0)*MY1 + 2*pow(KSx1,MP0)*MY1 +
          pow(2,1 + MP0)*MY2)*M_PI);
      XG[3] = Xks[3];

      if (XG[1] < startx[1] || XG[1] > stopx[1]) return 0;
    }

  } else {
    if(X[1] < startx[1] ||
       X[1] > stopx[1]  ||
       X[2] < startx[2] ||
       X[2] > stopx[2]) {
      return 0;
    }
  }

  return 1;
}

/*
 *  returns geodesic coordinates associated with center of zone i,j,k
 */
void ijktoX(int i, int j, int k, double X[NDIM]) 
{
  // first do the naive thing 
  X[1] = startx[1] + (i+0.5)*dx[1];
  X[2] = startx[2] + (j+0.5)*dx[2];
  X[3] = startx[3] + (k+0.5)*dx[3];

  // now transform to geodesic coordinates if necessary by first
  // converting to KS and then to destination coordinates (eKS).
  if (METRIC_eKS) {
      double xKS[4] = { 0 };
    if (METRIC_MKS3) {
      double x0 = X[0];
      double x1 = X[1];
      double x2 = X[2];
      double x3 = X[3];

      double H0 = mks3H0;
      double MY1 = mks3MY1;
      double MY2 = mks3MY2;
      double MP0 = mks3MP0;
      
      xKS[0] = x0;
      xKS[1] = exp(x1) + mks3R0;
      xKS[2] = (M_PI*(1+1./tan((H0*M_PI)/2.)*tan(H0*M_PI*(-0.5+(MY1+(pow(2,MP0)*(-MY1+MY2))/pow(exp(x1)+R0,MP0))*(1-2*x2)+x2))))/2.;
      xKS[3] = x3;
    }
    
    X[0] = xKS[0];
    X[1] = log(xKS[1]);
    X[2] = xKS[2] / M_PI;
    X[3] = xKS[3];
  }
}

/*
 *  translates geodesic coordinates to a grid zone and returns offset
 *  for interpolation purposes. integer index corresponds to the zone
 *  center "below" the desired point and del[i] \in [0,1) returns the
 *  offset from that zone center.
 *
 *  0    0.5    1
 *  [     |     ]
 *  A  B  C DE  F
 *
 *  startx = 0.
 *  dx = 0.5
 *
 *  A -> (-1, 0.5)
 *  B -> ( 0, 0.0)
 *  C -> ( 0, 0.5)
 *  D -> ( 0, 0.9)
 *  E -> ( 1, 0.0)
 *  F -> ( 1, 0.5)
 *
 */
void Xtoijk(double X[NDIM], int *i, int *j, int *k, double del[NDIM])
{
  // unless we're reading from data, i,j,k are the normal expected thing
  double phi;
  double XG[4];

  if (METRIC_eKS) {
    // the geodesics are evolved in eKS so invert through KS -> zone coordinates
    double Xks[4] = { X[0], exp(X[1]), M_PI*X[2], X[3] };
    if (METRIC_MKS3) {
      double H0 = mks3H0, MY1 = mks3MY1, MY2 = mks3MY2, MP0 = mks3MP0;
      double KSx1 = Xks[1], KSx2 = Xks[2];
      XG[0] = Xks[0];
      XG[1] = log(Xks[1] - mks3R0);
      XG[2] = (-(H0*pow(KSx1,MP0)*M_PI) - pow(2.,1. + MP0)*H0*MY1*M_PI + 
        2.*H0*pow(KSx1,MP0)*MY1*M_PI + pow(2.,1. + MP0)*H0*MY2*M_PI + 
        2.*pow(KSx1,MP0)*atan(((-2.*KSx2 + M_PI)*tan((H0*M_PI)/2.))/M_PI))/(2.*
        H0*(-pow(KSx1,MP0) - pow(2.,1 + MP0)*MY1 + 2.*pow(KSx1,MP0)*MY1 + 
          pow(2.,1. + MP0)*MY2)*M_PI);
      XG[3] = Xks[3];
    }
  } else {
    MULOOP XG[mu] = X[mu];
  }

  // the X[3] coordinate is allowed to vary so first map it to [0, stopx[3])
  phi = fmod(XG[3], stopx[3]);
  if(phi < 0.0) phi = stopx[3]+phi;

  // get provisional zone index. see note above function for details. note we
  // shift to zone centers because that's where variables are most exact.
  *i = (int) ((XG[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
  *j = (int) ((XG[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
  *k = (int) ((phi  - startx[3]) / dx[3] - 0.5 + 1000) - 1000;  

  // exotic coordinate systems sometime have issues. use this block to enforce
  // reasonable limits on *i,*j and *k. in the normal coordinate systems, this
  // block should never fire.
  if (*i < -1) *i = -1;
  if (*j < -1) *j = -1;
  if (*k < -1) *k = -1;
  if (*i >= N1) *i = N1-1;
  if (*j >= N2) *j = N2-1;
  if (*k >= N3) *k = N3-1;

  // now construct del
  del[1] = (XG[1] - ((*i + 0.5) * dx[1] + startx[1])) / dx[1];
  del[2] = (XG[2] - ((*j + 0.5) * dx[2] + startx[2])) / dx[2];
  del[3] = (phi - ((*k + 0.5) * dx[3] + startx[3])) / dx[3];

  // and enforce limits on del (for exotic coordinate systems)
  for (int i=0; i<4; ++i) {
    if (del[i] < 0.) del[i] = 0.;
    if (del[i] >= 1.) del[i] = 1.;
  }

}

//#define SINGSMALL (1.E-20)
/* return boyer-lindquist coordinate of point */
void bl_coord(double X[NDIM], double *r, double *th)
{
  *r = exp(X[1]);

  if (METRIC_eKS) {
    *r = exp(X[1]);
    *th = M_PI * X[2];
  } else if (METRIC_MKS3) {
    *r = exp(X[1]) + mks3R0;
    *th = (M_PI*(1. + 1./tan((mks3H0*M_PI)/2.)*tan(mks3H0*M_PI*(-0.5 + (mks3MY1 + (pow(2.,mks3MP0)*(-mks3MY1 + mks3MY2))/pow(exp(X[1])+mks3R0,mks3MP0))*(1. - 2.*X[2]) + X[2]))))/2.;
  } else if (DEREFINE_POLES) {
    double thG = M_PI*X[2] + ((1. - hslope)/2.)*sin(2.*M_PI*X[2]);
    double y = 2*X[2] - 1.;
    double thJ = poly_norm*y*(1. + pow(y/poly_xt,poly_alpha)/(poly_alpha+1.)) + 0.5*M_PI;
    *th = thG + exp(mks_smooth*(startx[1] - X[1]))*(thJ - thG);
  } else {
    *th = M_PI*X[2] + ((1. - hslope)/2.)*sin(2.*M_PI*X[2]);
  }
}

void set_units()
{
  L_unit = GNEWT * MBH / (CL * CL);
  T_unit = L_unit / CL;
  RHO_unit = M_unit / pow(L_unit, 3);
  U_unit = RHO_unit * CL * CL;
  B_unit = CL * sqrt(4.*M_PI*RHO_unit);
  Mdotedd=4.*M_PI*GNEWT*MBH*MP/CL/0.1/SIGMA_THOMSON;

  fprintf(stderr,"L,T,M units: %g [cm] %g [s] %g [g]\n",L_unit,T_unit,M_unit) ;
  fprintf(stderr,"rho,u,B units: %g [g cm^-3] %g [g cm^-1 s^-2] %g [G] \n",RHO_unit,U_unit,B_unit) ;
}

void init_physical_quantities(int n)
{
  // cover everything, even ghost zones
#pragma omp parallel for collapse(3)
  for (int i = 0; i < N1+2; i++) {
    for (int j = 0; j < N2+2; j++) {
      for (int k = 0; k < N3+2; k++) {
        data[n]->ne[i][j][k] = data[n]->p[KRHO][i][j][k] * RHO_unit/(MP+ME) ;

        double bsq = data[n]->bcon[i][j][k][0] * data[n]->bcov[i][j][k][0] +
              data[n]->bcon[i][j][k][1] * data[n]->bcov[i][j][k][1] +
              data[n]->bcon[i][j][k][2] * data[n]->bcov[i][j][k][2] +
              data[n]->bcon[i][j][k][3] * data[n]->bcov[i][j][k][3] ;

        data[n]->b[i][j][k] = sqrt(bsq)*B_unit;
        double sigma_m = bsq/data[n]->p[KRHO][i][j][k];

        if (ELECTRONS == 1) {
          data[n]->thetae[i][j][k] = data[n]->p[KEL][i][j][k]*pow(data[n]->p[KRHO][i][j][k],game-1.)*Thetae_unit;
        } else if (ELECTRONS == 2) {
          double beta = data[n]->p[UU][i][j][k]*(gam-1.)/0.5/bsq;
          double betasq = beta*beta;
          double trat = trat_large * betasq/(1. + betasq) + trat_small /(1. + betasq);
          //Thetae_unit = (gam - 1.) * (MP / ME) / trat;
          // see, e.g., Eq. 8 of the EHT GRRT formula list
          Thetae_unit = (MP/ME) * (game-1.) * (gamp-1.) / ( (gamp-1.) + (game-1.)*trat );
          data[n]->thetae[i][j][k] = Thetae_unit*data[n]->p[UU][i][j][k]/data[n]->p[KRHO][i][j][k];
        } else {
          data[n]->thetae[i][j][k] = Thetae_unit*data[n]->p[UU][i][j][k]/data[n]->p[KRHO][i][j][k];
        }
        data[n]->thetae[i][j][k] = MAX(data[n]->thetae[i][j][k], 1.e-3);
       
        //thetae[i][j][k] = (gam-1.)*MP/ME*p[UU][i][j][k]/p[KRHO][i][j][k];
        //printf("rho = %e thetae = %e\n", p[KRHO][i][j][k], thetae[i][j][k]);

        //strongly magnetized = empty, no shiny spine
        if (sigma_m > SIGMA_CUT) {
          data[n]->b[i][j][k]=0.0;
          data[n]->ne[i][j][k]=0.0;
          data[n]->thetae[i][j][k]=0.0;
        }
      }
    }
  }

}

// malloc utilities
void *malloc_rank1(int n1, int size)
{
  void *A;

  if ((A = malloc(n1*size)) == NULL) {
    fprintf(stderr,"malloc failure in malloc_rank1\n");
    exit(123);
  }

  return A;
}

double **malloc_rank2(int n1, int n2)
{

  double **A;
  double *space;
  int i;

  space = malloc_rank1(n1*n2, sizeof(double));
  A = malloc_rank1(n1, sizeof(double *));
  for(i = 0; i < n1; i++) A[i] = &(space[i*n2]);

  return A;
}


double ***malloc_rank3(int n1, int n2, int n3)
{

  double ***A;
  double *space;
  int i,j;

  space = malloc_rank1(n1*n2*n3, sizeof(double));
  A = malloc_rank1(n1, sizeof(double *));
  for(i = 0; i < n1; i++){
    A[i] = malloc_rank1(n2,sizeof(double *));
    for(j = 0; j < n2; j++){
      A[i][j] = &(space[n3*(j + n2*i)]);
    }
  }

  return A;
}

float **malloc_rank2_float(int n1, int n2)
{

  float **A;
  float *space;
  int i;

  space = malloc_rank1(n1*n2, sizeof(float));
  A = malloc_rank1(n1, sizeof(float *));
  for(i = 0; i < n1; i++) A[i] = &(space[i*n2]);

  return A;
}


float ***malloc_rank3_float(int n1, int n2, int n3)
{

  float ***A;
  float *space;
  int i,j;

  space = malloc_rank1(n1*n2*n3, sizeof(float));
  A = malloc_rank1(n1, sizeof(float *));
  for(i = 0; i < n1; i++){
    A[i] = malloc_rank1(n2,sizeof(float *));
    for(j = 0; j < n2; j++){
      A[i][j] = &(space[n3*(j + n2*i)]);
    }
  }

  return A;
}

float ****malloc_rank4_float(int n1, int n2, int n3, int n4)
{

  float ****A;
  float *space;
  int i,j,k;

  space = malloc_rank1(n1*n2*n3*n4, sizeof(float));
  A = malloc_rank1(n1, sizeof(float *));
  for(i=0;i<n1;i++){
    A[i] = malloc_rank1(n2,sizeof(float *));
    for(j=0;j<n2;j++){
      A[i][j] = malloc_rank1(n3,sizeof(float *));
      for(k=0;k<n3;k++){
        A[i][j][k] = &(space[n4*(k + n3*(j + n2*i))]);
      }
    }
  }

  return A;
}


double ****malloc_rank4(int n1, int n2, int n3, int n4)
{

  double ****A;
  double *space;
  int i,j,k;

  space = malloc_rank1(n1*n2*n3*n4, sizeof(double));
  A = malloc_rank1(n1, sizeof(double *));
  for(i=0;i<n1;i++){
    A[i] = malloc_rank1(n2,sizeof(double *));
    for(j=0;j<n2;j++){
      A[i][j] = malloc_rank1(n3,sizeof(double *));
      for(k=0;k<n3;k++){
        A[i][j][k] = &(space[n4*(k + n3*(j + n2*i))]);
      }
    }
  }

  return A;
}

double *****malloc_rank5(int n1, int n2, int n3, int n4, int n5)
{

  double *****A;
  double *space;
  int i,j,k,l;

  space = malloc_rank1(n1*n2*n3*n4*n5, sizeof(double));
  A = malloc_rank1(n1, sizeof(double *));
  for(i=0;i<n1;i++){
    A[i] = malloc_rank1(n2, sizeof(double *));
    for(j=0;j<n2;j++){
      A[i][j] = malloc_rank1(n3, sizeof(double *));
      for(k=0;k<n3;k++){
        A[i][j][k] = malloc_rank1(n4, sizeof(double *));
        for(l=0;l<n4;l++){
          A[i][j][k][l] = &(space[n5*(l + n4*(k + n3*(j + n2*i)))]);
        }
      }
    }
  }

  return A;
}

void init_storage(void)
{
  // one ghost zone on each side of the domain
  for (int n = 0; n < NSUP; n++) {
    data[n]->bcon = malloc_rank4(N1+2,N2+2,N3+2,NDIM);
    data[n]->bcov = malloc_rank4(N1+2,N2+2,N3+2,NDIM);
    data[n]->ucon = malloc_rank4(N1+2,N2+2,N3+2,NDIM);
    data[n]->ucov = malloc_rank4(N1+2,N2+2,N3+2,NDIM);
    data[n]->p = malloc_rank4(NVAR,N1+2,N2+2,N3+2);
    //p = (double ****)malloc_rank1(NVAR,sizeof(double *));
    //for(i = 0; i < NVAR; i++) p[i] = malloc_rank3(N1,N2,N3);
    data[n]->ne = malloc_rank3(N1+2,N2+2,N3+2);
    data[n]->thetae = malloc_rank3(N1+2,N2+2,N3+2);
    data[n]->b = malloc_rank3(N1+2,N2+2,N3+2);
  }
}

/* HDF5 v1.8 API */
#include <hdf5.h>
#include <hdf5_hl.h>

void init_iharm_grid(char *fnam, int dumpidx)
{
  // called at the beginning of the run and sets the static parameters
  // along with setting up the grid
  
  char fname[256];
  snprintf(fname, 255, fnam, dumpidx);
 
  fprintf(stderr, "filename: %s\n", fname);

  fprintf(stderr, "init grid\n");

  if ( hdf5_open(fname) < 0 ) {
    fprintf(stderr, "! unable to open file %s. exiting!\n", fname);
    exit(-2);
  }

  // get dump info to copy to ipole output
  hdf5_read_single_val(&t0, "t", H5T_IEEE_F64LE);
  fluid_header = hdf5_get_blob("/header");

  hdf5_set_directory("/header/");

  if ( hdf5_exists("has_electrons") )
    hdf5_read_single_val(&ELECTRONS, "has_electrons", H5T_STD_I32LE);
  if ( hdf5_exists("has_radiation") ) 
    hdf5_read_single_val(&RADIATION, "has_radiation", H5T_STD_I32LE);
  if ( hdf5_exists("has_derefine_poles") )
    hdf5_read_single_val(&DEREFINE_POLES, "has_derefine_poles", H5T_STD_I32LE);

  char metric[20];
  hid_t HDF5_STR_TYPE = hdf5_make_str_type(20);
  hdf5_read_single_val(&metric, "metric", HDF5_STR_TYPE);

  DEREFINE_POLES = 0;
  METRIC_MKS3 = 0;

  if ( strncmp(metric, "MMKS", 19) == 0 ) {
    DEREFINE_POLES = 1;
  } else if ( strncmp(metric, "MKS3", 19) == 0 ) {
    METRIC_eKS = 1;
    METRIC_MKS3 = 1;
    fprintf(stderr, "using eKS metric with exotic \"MKS3\" zones...\n");
  }

  hdf5_read_single_val(&N1, "n1", H5T_STD_I32LE);
  hdf5_read_single_val(&N2, "n2", H5T_STD_I32LE);
  hdf5_read_single_val(&N3, "n3", H5T_STD_I32LE);
  hdf5_read_single_val(&gam, "gam", H5T_IEEE_F64LE);

  if (hdf5_exists("gam_e")) {
    fprintf(stderr, "custom electron model loaded from dump file...\n");
    hdf5_read_single_val(&game, "gam_e", H5T_IEEE_F64LE);
    hdf5_read_single_val(&gamp, "gam_p", H5T_IEEE_F64LE);
    Thetae_unit = MP/ME;
  }
  Te_unit = Thetae_unit;

  // we can override which electron model to use here. print results if we're
  // overriding anything. ELECTRONS should only be nonzero if we need to make
  // use of extra variables (instead of just UU and RHO) for thetae
  if (!USE_FIXED_TPTE && !USE_MIXED_TPTE) {
    if (ELECTRONS != 1) {
      fprintf(stderr, "! no electron temperature model specified in model/iharm.c\n");
      exit(-3);
    }
    ELECTRONS = 1;
  } else if (USE_FIXED_TPTE && !USE_MIXED_TPTE) {
    ELECTRONS = 0; // force TP_OVER_TE to overwrite bad electrons
    fprintf(stderr, "using fixed tp_over_te ratio = %g\n", tp_over_te);
    //Thetae_unit = MP/ME*(gam-1.)*1./(1. + tp_over_te);
    // see, e.g., Eq. 8 of the EHT GRRT formula list. 
    // this formula assumes game = 4./3 and gamp = 5./3
    Thetae_unit = 2./3. * MP/ME / (2. + tp_over_te);
  } else if (USE_MIXED_TPTE && !USE_FIXED_TPTE) {
    ELECTRONS = 2;
    fprintf(stderr, "using mixed tp_over_te with trat_small = %g and trat_large = %g\n", trat_small, trat_large);
  } else {
    fprintf(stderr, "! please change electron model in model/iharm.c\n");
    exit(-3);
  }

  // by this point, we're sure that Thetae_unit is what we want so we can set
  // Te_unit which is what ultimately get written to the dump files
  Te_unit = Thetae_unit;

  if (RADIATION) {
    fprintf(stderr, "custom radiation field tracking information loaded...\n");
    fprintf(stderr, "!! warning, this branch is not tested!\n");
    hdf5_set_directory("/header/units/");
    hdf5_read_single_val(&M_unit, "M_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&T_unit, "T_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&L_unit, "L_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&Thetae_unit, "Thetae_unit", H5T_IEEE_F64LE);
    hdf5_read_single_val(&MBH, "Mbh", H5T_IEEE_F64LE);
    hdf5_read_single_val(&tp_over_te, "tp_over_te", H5T_IEEE_F64LE);
  }

  hdf5_set_directory("/header/geom/");
  hdf5_read_single_val(&startx[1], "startx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[2], "startx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&startx[3], "startx3", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[1], "dx1", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[2], "dx2", H5T_IEEE_F64LE);
  hdf5_read_single_val(&dx[3], "dx3", H5T_IEEE_F64LE);

  hdf5_set_directory("/header/geom/mks/");
  if ( DEREFINE_POLES ) hdf5_set_directory("/header/geom/mmks/");
  if ( METRIC_MKS3 ) {
    hdf5_set_directory("/header/geom/mks3/");
    hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3R0, "R0", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3H0, "H0", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MY1, "MY1", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MY2, "MY2", H5T_IEEE_F64LE);
    hdf5_read_single_val(&mks3MP0, "MP0", H5T_IEEE_F64LE);
    Rout = 100.; 
  } else {
    hdf5_read_single_val(&a, "a", H5T_IEEE_F64LE);
    hdf5_read_single_val(&hslope, "hslope", H5T_IEEE_F64LE);
    if (hdf5_exists("Rin")) {
      hdf5_read_single_val(&Rin, "Rin", H5T_IEEE_F64LE);
      hdf5_read_single_val(&Rout, "Rout", H5T_IEEE_F64LE);
    } else {
      hdf5_read_single_val(&Rin, "r_in", H5T_IEEE_F64LE);
      hdf5_read_single_val(&Rout, "r_out", H5T_IEEE_F64LE);
    }

    if (DEREFINE_POLES) {
      fprintf(stderr, "custom refinement at poles loaded...\n");
      hdf5_read_single_val(&poly_xt, "poly_xt", H5T_IEEE_F64LE);
      hdf5_read_single_val(&poly_alpha, "poly_alpha", H5T_IEEE_F64LE);
      hdf5_read_single_val(&mks_smooth, "mks_smooth", H5T_IEEE_F64LE);
      poly_norm = 0.5*M_PI*1./(1. + 1./(poly_alpha + 1.)*1./pow(poly_xt, poly_alpha));
    }
  }
 
  rmax_geo = MIN(100., Rout);

  hdf5_set_directory("/");
  hdf5_read_single_val(&DTd, "dump_cadence", H5T_IEEE_F64LE);
  
  //th_beg=th_cutout;
  //th_end=M_PI-th_cutout;
  //th_len = th_end-th_beg;

  // Ignore radiation interactions within one degree of polar axis
  th_beg = 0.0174;
  //th_end = 3.1241;

  stopx[0] = 1.;
  stopx[1] = startx[1]+N1*dx[1];
  stopx[2] = startx[2]+N2*dx[2];
  stopx[3] = startx[3]+N3*dx[3];

  fprintf(stderr, "start: %g %g %g \n", startx[1], startx[2], startx[3]);
  fprintf(stderr, "stop: %g %g %g \n", stopx[1], stopx[2], stopx[3]);

  init_storage();

  hdf5_close();
}

void output_hdf5(hid_t fid)
{
#if SLOW_LIGHT
  h5io_add_data_dbl(fid, "/header/t", data[1]->t); 
#else // FAST LIGHT
  h5io_add_data_dbl(fid, "/header/t", data[0]->t); 
#endif
  h5io_add_blob(fid, "/fluid_header", fluid_header); 

  h5io_add_group(fid, "/header/electrons");
  if (ELECTRONS == 0) {
    h5io_add_data_dbl(fid, "/header/electrons/tp_over_te", tp_over_te);
  } else if (ELECTRONS == 2) {
    h5io_add_data_dbl(fid, "/header/electrons/rlow", trat_small);
    h5io_add_data_dbl(fid, "/header/electrons/rhigh", trat_large);
  }
  h5io_add_data_int(fid, "/header/electrons/type", ELECTRONS);
}

void load_iharm_data(int n, char *fnam, int dumpidx, int verbose)
{
  // loads relevant information from fluid dump file stored at fname
  // to the n'th copy of data (e.g., for slow light)

  double dMact, Ladv;

  char fname[256];
  snprintf(fname, 255, fnam, dumpidx);

  if (verbose) fprintf(stderr, "LOADING DATA\n");
  nloaded++;

  if ( hdf5_open(fname) < 0 ) {
    fprintf(stderr, "! unable to open file %s. Exiting!\n", fname);
    exit(-1);
  }

  hdf5_set_directory("/");

  int n_prims;
  hdf5_read_single_val(&n_prims, "/header/n_prim", H5T_STD_I32LE);

  // load into "center" of data
  hsize_t fdims[] = { N1, N2, N3, n_prims };
  hsize_t fstart[] = { 0, 0, 0, 0 };
  hsize_t fcount[] = { N1, N2, N3, 1 };
  hsize_t mdims[] = { N1+2, N2+2, N3+2, 1 };
  hsize_t mstart[] = { 1, 1, 1, 0 };

  fstart[3] = 0;
  hdf5_read_array(data[n]->p[KRHO][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 1;
  hdf5_read_array(data[n]->p[UU][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 2;
  hdf5_read_array(data[n]->p[U1][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 3;
  hdf5_read_array(data[n]->p[U2][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 4;
  hdf5_read_array(data[n]->p[U3][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 5;
  hdf5_read_array(data[n]->p[B1][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 6;
  hdf5_read_array(data[n]->p[B2][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  fstart[3] = 7;
  hdf5_read_array(data[n]->p[B3][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE); 

  if (ELECTRONS == 1) {
    fstart[3] = 8;
    hdf5_read_array(data[n]->p[KEL][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
    fstart[3] = 9;
    hdf5_read_array(data[n]->p[KTOT][0][0], "prims", 4, fdims, fstart, fcount, mdims, mstart, H5T_IEEE_F64LE);
  }

  hdf5_read_single_val(&(data[n]->t), "t", H5T_IEEE_F64LE);

  hdf5_close();

  dMact = Ladv = 0.;

  // construct four-vectors over "real" zones
#pragma omp parallel for collapse(2) reduction(+:dMact,Ladv)
  for(int i = 1; i < N1+1; i++) {
    for(int j = 1; j < N2+1; j++) {

      double X[NDIM] = { 0. };
      double gcov[NDIM][NDIM], gcon[NDIM][NDIM];
      double g, r, th;

      // this assumes axisymmetry in the coordinates
      ijktoX(i-1,j-1,0, X);
      gcov_func(X, gcov);
      gcon_func(gcov, gcon);
      g = gdet_zone(i-1,j-1,0);

      bl_coord(X, &r, &th);

      for(int k = 1; k < N3+1; k++){

        ijktoX(i-1,j-1,k,X);
        double UdotU = 0.;
        
        // the four-vector reconstruction should have gcov and gcon and gdet using the modified coordinates
        // interpolating the four vectors to the zone center !!!!
        for(int l = 1; l < NDIM; l++) 
          for(int m = 1; m < NDIM; m++) 
            UdotU += gcov[l][m]*data[n]->p[U1+l-1][i][j][k]*data[n]->p[U1+m-1][i][j][k];
        double ufac = sqrt(-1./gcon[0][0]*(1 + fabs(UdotU)));
        data[n]->ucon[i][j][k][0] = -ufac*gcon[0][0];
        for(int l = 1; l < NDIM; l++) 
          data[n]->ucon[i][j][k][l] = data[n]->p[U1+l-1][i][j][k] - ufac*gcon[0][l];
        lower(data[n]->ucon[i][j][k], gcov, data[n]->ucov[i][j][k]);

        // reconstruct the magnetic field three vectors
        double udotB = 0.;
        
        for (int l = 1; l < NDIM; l++) {
          udotB += data[n]->ucov[i][j][k][l]*data[n]->p[B1+l-1][i][j][k];
        }
        
        data[n]->bcon[i][j][k][0] = udotB;

        for (int l = 1; l < NDIM; l++) {
          data[n]->bcon[i][j][k][l] = (data[n]->p[B1+l-1][i][j][k] + data[n]->ucon[i][j][k][l]*udotB)/data[n]->ucon[i][j][k][0];
        }

        lower(data[n]->bcon[i][j][k], gcov, data[n]->bcov[i][j][k]);

        if(i <= 21) { dMact += g * data[n]->p[KRHO][i][j][k] * data[n]->ucon[i][j][k][1]; }
        if(i >= 21 && i < 41 && 0) Ladv += g * data[n]->p[UU][i][j][k] * data[n]->ucon[i][j][k][1] * data[n]->ucov[i][j][k][0] ;
        if(i <= 21) Ladv += g * data[n]->p[UU][i][j][k] * data[n]->ucon[i][j][k][1] * data[n]->ucov[i][j][k][0] ;

      }
    }
  }

  // now copy primitives and four-vectors according to boundary conditions

  // radial -- just extend zones
#pragma omp parallel for collapse(2)
  for (int j=1; j<N2+1; ++j) {
    for (int k=1; k<N3+1; ++k) {
      for (int l=0; l<NDIM; ++l) {
        data[n]->bcon[0][j][k][l] = data[n]->bcon[1][j][k][l];
        data[n]->bcon[N1+1][j][k][l] = data[n]->bcon[N1][j][k][l];
        data[n]->bcov[0][j][k][l] = data[n]->bcov[1][j][k][l];
        data[n]->bcov[N1+1][j][k][l] = data[n]->bcov[N1][j][k][l];
        data[n]->ucon[0][j][k][l] = data[n]->ucon[1][j][k][l];
        data[n]->ucon[N1+1][j][k][l] = data[n]->ucon[N1][j][k][l];
        data[n]->ucov[0][j][k][l] = data[n]->ucov[1][j][k][l];
        data[n]->ucov[N1+1][j][k][l] = data[n]->ucov[N1][j][k][l];
      }
      for (int l=0; l<NVAR; ++l) {
        data[n]->p[l][0][j][k] = data[n]->p[l][1][j][k];
        data[n]->p[l][N1+1][j][k] = data[n]->p[l][N1][j][k];
      }
    }
  }

  // elevation -- flip (this is a rotation by pi)
#pragma omp parallel for collapse(2)
  for (int i=0; i<N1+2; ++i) {
    for (int k=1; k<N3+1; ++k) {
      if (N3%2 == 0) {
        int kflip = ( k + (N3/2) ) % N3;
        for (int l=0; l<NDIM; ++l) {
          data[n]->bcon[i][0][k][l] = data[n]->bcon[i][1][kflip][l];
          data[n]->bcon[i][N2+1][k][l] = data[n]->bcon[i][N2][kflip][l];
          data[n]->bcov[i][0][k][l] = data[n]->bcov[i][1][kflip][l];
          data[n]->bcov[i][N2+1][k][l] = data[n]->bcov[i][N2][kflip][l];
          data[n]->ucon[i][0][k][l] = data[n]->ucon[i][1][kflip][l];
          data[n]->ucon[i][N2+1][k][l] = data[n]->ucon[i][N2][kflip][l];
          data[n]->ucov[i][0][k][l] = data[n]->ucov[i][1][kflip][l];
          data[n]->ucov[i][N2+1][k][l] = data[n]->ucov[i][N2][kflip][l];
        }
        for (int l=0; l<NVAR; ++l) {
          data[n]->p[l][i][0][k] = data[n]->p[l][i][1][kflip];
          data[n]->p[l][i][N2+1][k] = data[n]->p[l][i][N2][kflip];
        }
      } else {
        int kflip1 = ( k + (N3/2) ) % N3;
        int kflip2 = ( k + (N3/2) + 1 ) % N3;
        for (int l=0; l<NDIM; ++l) {
          data[n]->bcon[i][0][k][l]    = ( data[n]->bcon[i][1][kflip1][l] 
                                         + data[n]->bcon[i][1][kflip2][l] ) / 2.;
          data[n]->bcon[i][N2+1][k][l] = ( data[n]->bcon[i][N2][kflip1][l]
                                         + data[n]->bcon[i][N2][kflip2][l] ) / 2.;
          data[n]->bcov[i][0][k][l]    = ( data[n]->bcov[i][1][kflip1][l]
                                         + data[n]->bcov[i][1][kflip2][l] ) / 2.;
          data[n]->bcov[i][N2+1][k][l] = ( data[n]->bcov[i][N2][kflip1][l] 
                                         + data[n]->bcov[i][N2][kflip2][l] ) / 2.;
          data[n]->ucon[i][0][k][l]    = ( data[n]->ucon[i][1][kflip1][l]
                                         + data[n]->ucon[i][1][kflip2][l] ) / 2.;
          data[n]->ucon[i][N2+1][k][l] = ( data[n]->ucon[i][N2][kflip1][l]
                                         + data[n]->ucon[i][N2][kflip2][l] ) / 2.;
          data[n]->ucov[i][0][k][l]    = ( data[n]->ucov[i][1][kflip1][l] 
                                         + data[n]->ucov[i][1][kflip2][l] ) / 2.;
          data[n]->ucov[i][N2+1][k][l] = ( data[n]->ucov[i][N2][kflip1][l] 
                                         + data[n]->ucov[i][N2][kflip2][l] ) / 2.;
        }
        for (int l=0; l<NVAR; ++l) {
          data[n]->p[l][i][0][k]    = ( data[n]->p[l][i][1][kflip1]
                                      + data[n]->p[l][i][1][kflip2] ) / 2.;
          data[n]->p[l][i][N2+1][k] = ( data[n]->p[l][i][N2][kflip1]
                                      + data[n]->p[l][i][N2][kflip2] ) / 2.;
        }
      }
    }
  }

  // azimuth -- periodic
#pragma omp parallel for collapse(2)
  for (int i=0; i<N1+2; ++i) {
    for (int j=0; j<N2+2; ++j) {
      for (int l=0; l<NDIM; ++l) {
        data[n]->bcon[i][j][0][l] = data[n]->bcon[i][j][N3][l];
        data[n]->bcon[i][j][N3+1][l] = data[n]->bcon[i][j][1][l];
        data[n]->bcov[i][j][0][l] = data[n]->bcov[i][j][N3][l];
        data[n]->bcov[i][j][N3+1][l] = data[n]->bcov[i][j][1][l];
        data[n]->ucon[i][j][0][l] = data[n]->ucon[i][j][N3][l];
        data[n]->ucon[i][j][N3+1][l] = data[n]->ucon[i][j][1][l];
        data[n]->ucov[i][j][0][l] = data[n]->ucov[i][j][N3][l];
        data[n]->ucov[i][j][N3+1][l] = data[n]->ucov[i][j][1][l];
      }
      for (int l=0; l<NVAR; ++l) {
        data[n]->p[l][i][j][0] = data[n]->p[l][i][j][N3];
        data[n]->p[l][i][j][N3+1] = data[n]->p[l][i][j][1];
      }
    }
  }

  dMact *= dx[3]*dx[2] ;
  dMact /= 21. ;
  Ladv *= dx[3]*dx[2] ;
  Ladv /= 21. ;

  if (verbose) {
    fprintf(stderr,"dMact: %g [code]\n",dMact) ;
    fprintf(stderr,"Ladv: %g [code]\n",Ladv) ;
    fprintf(stderr,"Mdot: %g [g/s] \n",-dMact*M_unit/T_unit) ;
    fprintf(stderr,"Mdot: %g [MSUN/YR] \n",-dMact*M_unit/T_unit/(MSUN / YEAR)) ;
    fprintf(stderr,"Mdot: %g [Mdotedd]\n",-dMact*M_unit/T_unit/Mdotedd) ;
    fprintf(stderr,"Mdotedd: %g [g/s]\n",Mdotedd) ;
    fprintf(stderr,"Mdotedd: %g [MSUN/YR]\n",Mdotedd/(MSUN/YEAR)) ;
  }

  // now construct useful scalar quantities (over full (+ghost) zones of data)
  init_physical_quantities(n);

  // optionally calculate average beta weighted by jnu
  if (0 == 1) {
    #define NBETABINS 64.
    double betabins[64] = { 0 };
    #define BETAMIN (0.001)
    #define BETAMAX (200.)
    double dlBeta = (log(BETAMAX)-log(BETAMIN))/NBETABINS;
    double BETA0 = log(BETAMIN);
    double betajnugdet = 0.;
    double jnugdet = 0.;
    for (int i=1; i<N1+1; ++i) {
      for (int j=1; j<N2+1; ++j) {
        for (int k=1; k<N3+1; ++k) {
          int zi = i-1; 
          int zj = j-1;
          int zk = k-1;
          double bsq = 0.;
          for (int l=0; l<4; ++l) bsq += data[n]->bcon[i][j][k][l]*data[n]->bcov[i][j][k][l];
          double beta = data[n]->p[UU][i][j][k]*(gam-1.)/0.5/bsq;
          double Ne = data[n]->ne[i][j][k];
          double Thetae = data[n]->thetae[i][j][k];
          double B = data[n]->b[i][j][k];
          double jnu = jnu_synch(2.3e+11, Ne, Thetae, B, M_PI/3.);
          double gdetzone = gdet_zone(zi,zj,zk);
          betajnugdet += beta * jnu * gdetzone;
          jnugdet += jnu * gdetzone;
          int betai = (int) ( (log(beta) - BETA0) / dlBeta + 2.5 ) - 2;
          betabins[betai] += jnu * gdetzone;
        }
      }
    }
    for (int i=0; i<NBETABINS; ++i) {
      fprintf(stderr, "%d %g %g\n", i, exp(BETA0 + (i+0.5)*dlBeta), betabins[i]);
    }
    fprintf(stderr, "<beta> = %g\n", betajnugdet / jnugdet);
  }
}

double root_find(double x[NDIM])
{
    double th = x[2];
    double thb, thc;
    double dtheta_func(double X[NDIM]), theta_func(double X[NDIM]);

    double Xa[NDIM], Xb[NDIM], Xc[NDIM];
    Xa[1] = log(x[1]);
    Xa[3] = x[3];
    Xb[1] = Xa[1];
    Xb[3] = Xa[3];
    Xc[1] = Xa[1];
    Xc[3] = Xa[3];

    if (x[2] < M_PI / 2.) {
      Xa[2] = 0. - SMALL;
      Xb[2] = 0.5 + SMALL;
    } else {
      Xa[2] = 0.5 - SMALL;
      Xb[2] = 1. + SMALL;
    }

    //tha = theta_func(Xa);
    thb = theta_func(Xb);

    /* bisect for a bit */
    double tol = 1.e-6;
    for (int i = 0; i < 100; i++) {
      Xc[2] = 0.5 * (Xa[2] + Xb[2]);
      thc = theta_func(Xc);

      if ((thc - th) * (thb - th) < 0.)
        Xa[2] = Xc[2];
      else
        Xb[2] = Xc[2];

      double err = theta_func(Xc) - th;
      if (fabs(err) < tol) break;
    }

    return (Xa[2]);
}

/*this does not depend on theta cut-outs there is no squizzing*/
double theta_func(double X[NDIM])
{
  double r, th;
  bl_coord(X, &r, &th);
  return th;
}

int radiating_region(double X[NDIM])
{
  if (X[1] < log(rmax_geo) && X[2]>th_beg/M_PI && X[2]<(1.-th_beg/M_PI) ) {
    return 1;
  } else {
    return 0;
  }
}

