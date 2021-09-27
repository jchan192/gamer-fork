#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

GPU_DEVICE static
void findtoreps( const real x, const real y, const real z,
                 real *found_lt, const real *table_Aux,
                 const int nx, const int ny, const int nz, const int ntemp,
                 const real *xt, const real *yt, const real *zt, const real *logtoreps,
                 const int IntScheme_Aux, int *keyerr );

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
// Description :  Finding temperature of internal energy from different modes
//                --> Energy      mode (0)
//                    Temperature mode (1)
//                    Entropy     mode (2)
//                    Pressure    mode (3)
//
// Note        :  1. Use 3D Lagrange linear interpolation or Catmull-Rom cubic interpolation formulae
//                   to search the corresponding temperature/energy given (rho, eps/temp/entr/pres, Ye)
//                2. Invoked by nuc_eos_C_short()
//
// Parameter   :  x             : Input vector of first  variable (rho)
//                y             : Input vector of second variable (eps/temp/entr/pres)
//                z             : Input vector of third  variable (Ye)
//                found_ltoreps : Output log(temp)/log(eps) of interpolated function values
//                table_Aux     : 3D array of tabulated logtemp/logenergy
//                nx            : X-dimension of table
//                ny            : Y-dimension of table
//                nz            : Z-dimension of table
//                ntoreps       : Size of temperature/energy array in the Nuclear EoS table
//                xt            : Vector of x-coordinates of table
//                yt            : Vector of y-coordinates of table
//                zt            : Vector of z-coordinates of table
//                logtoreps     : log(temp)/log(eps) array in the table
//                IntScheme_Aux : Interpolation scheme for the auxiliary table
//                keyerr        : Output error
//
// Return      :  found_ltoreps
//-------------------------------------------------------------------------------------
GPU_DEVICE
void findtoreps( const real x, const real y, const real z,
                 real *found_ltoreps, const real *table_Aux,
                 const int nx, const int ny, const int nz, const int ntoreps,
                 const real *xt, const real *yt, const real *zt, const real *logtoreps,
                 const int IntScheme_Aux, int *keyerr )
{

   int  TargetIdx[1] = { 0 };
   real output_vars[1];


   if ( IntScheme_Aux == NUC_INT_LINEAR )
   {
      nuc_eos_C_linterp_some(x, y, z, TargetIdx, output_vars, table_Aux,
                             nx, ny, nz, 1, xt, yt, zt);
   }

   else
   {
      nuc_eos_C_cubinterp_some(x, y, z, TargetIdx, output_vars, table_Aux,
                               nx, ny, nz, 1, xt, yt, zt);
   }


   *found_ltoreps = output_vars[0];

   if ( *found_ltoreps != *found_ltoreps  ||
        ! ( *found_ltoreps > logtoreps[0]  &&  *found_ltoreps < logtoreps[ntoreps-1] )  )
      *keyerr = 665;

} // FUNCTION : findtoreps



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
