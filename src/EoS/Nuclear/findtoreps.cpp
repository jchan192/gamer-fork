#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

GPU_DEVICE static
void findtoreps( const real x, const real y, const real z,
                 real *found_lt, const real *alltables_mode,
                 const int nx, const int ny, const int nz, const int ntemp,
                 const real *xt, const real *yt, const real *zt, const real *logtoreps,
                 const int interpol_TL, const int keymode, int *keyerr );

GPU_DEVICE static
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );

GPU_DEVICE static
void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );

#else

void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );

void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );

#endif // #ifdef __CUDACC__ ... else ...




//-------------------------------------------------------------------------------------
// Function    :  findtoreps
// Description :  Finding temperature from different modes
//                -->                 energy mode   (0)
//                                    entropy mode  (2)
//                                    pressure mode (3)
//
// Note        :  1. Use 3D Catmull-Rom cubic interpolation formula
//                   to search the corresponding temperature/energy given
//                   (rho, (eps, T, entropy, P), Y_e)
//                2. Invoked by nuc_eos_C_short()
//
// Parameter   :  x              : Input vector of first  variable (rho)
//                y              : Input vector of second variable (eps, T, entropy, P)
//                z              : Input vector of third  variable (Y_e)
//                found_ltoreps  : Output log(temp)/log(energy) of interpolated function values
//                alltables_mode : 3d array of tabulated logtemp
//                nx             : X-dimension of table
//                ny             : Y-dimension of table
//                nz             : Z-dimension of table
//                ntoreps        : Size of (temperature/energy) array in the Nuclear Eos table
//                xt             : Vector of x-coordinates of table
//                yt             : Vector of y-coordinates of table
//                zt             : Vector of z-coordinates of table
//                logtoreps      : log(T)/log(energy) array in the table
//                interpol_TL    : interpolation schemes for table look-ups (linear/cubic)
//                keymode        : Which mode we will use
//                                 --> 1: Energy mode   (coming in with internal energy)
//                                     2: Entropy mode  (coming in with entropy)
//                                     3: Pressure mode (coming in with P)
//                keyerr         : Output error
//
// Return      :  found_ltoreps
//-------------------------------------------------------------------------------------
GPU_DEVICE
void findtoreps( const real x, const real y, const real z,
                 real *found_ltoreps, const real *alltables_mode,
                 const int nx, const int ny, const int nz, const int ntoreps,
                 const real *xt, const real *yt, const real *zt, const real *logtoreps,
                 const int interpol_TL, const int keymode, int *keyerr )
{

   int  TargetIdx[1];
   real output_vars[1];

   switch ( keymode )
   {
      case NUC_MODE_PRES :   TargetIdx[0] = NUC_VAR_IDX_PRES;   break;

#     if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
      case NUC_MODE_ENGY :   TargetIdx[0] = NUC_VAR_IDX_EORT;   break;
#     else
      case NUC_MODE_TEMP :   TargetIdx[0] = NUC_VAR_IDX_EORT;   break;
#     endif // if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )

      case NUC_MODE_ENTR :   TargetIdx[0] = NUC_VAR_IDX_ENTR;   break;
   }


   if ( interpol_TL == NUC_INTERPOL_LINEAR )
   {
      nuc_eos_C_linterp_some(x, y, z, TargetIdx, output_vars, alltables_mode,
                             nx, ny, nz, 1, xt, yt, zt);
   }

   else
   {
      nuc_eos_C_cubinterp_some(x, y, z, TargetIdx, output_vars, alltables_mode,
                               nx, ny, nz, 1, xt, yt, zt);
   }


   *found_ltoreps = output_vars[0];

   if ( *found_ltoreps != *found_ltoreps  ||
        ! ( *found_ltoreps > logtoreps[0]  &&  *found_ltoreps < logtoreps[ntoreps-1] )  )
      *keyerr = 665;

} // FUNCTION : findtoreps



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
