#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__
GPU_DEVICE static
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );
#endif




//-------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_linterp_some
// Description :  Find thermodynamic variables using linear interpolation by searching
//                the tabulated nuclear EoS
//
// Note        :  1. Invoked by nuc_eos_C_short() and nuc_eos_C_cubinterp_some()
//
// Parameter   :  x           : Input vector of first  variable (rho)
//                y           : Input vector of second variable (eps/temp/entr/pres)
//                z           : Input vector of third  variable (Ye)
//                TargetIdx   : Indices of thermodynamic variables to be found
//                output_vars : Output variables of interpolated function values
//                alltables   : 3D array of tabulated variables
//                nx          : X-dimension of table
//                ny          : Y-dimension of table
//                nz          : Z-dimension of table
//                nvars       : Number of variables we will find
//                xt          : Vector of x-coordinates of table
//                yt          : Vector of y-coordinates of table
//                zt          : Vector of z-coordinates of table
//
// Return      :  output_vars
//-------------------------------------------------------------------------------------
GPU_DEVICE
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             const int *TargetIdx, real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt )
{

// helper variables
   real delx, dely, delz, a[8], fh[8];
   real dx, dy, dz, dxi, dyi, dzi, dxyi, dxzi, dyzi, dxyzi;
   int  ix, iy, iz, iv, idx;
   int  nxy  = nx*ny;
   int  nxyz = nxy*nz;


// determine spacing parameters of equidistant (!!!) table
   dx    = ( xt[nx-1] - xt[0] ) / (real)(nx-1);
   dy    = ( yt[ny-1] - yt[0] ) / (real)(ny-1);
   dz    = ( zt[nz-1] - zt[0] ) / (real)(nz-1);

   dxi   = (real)1.0 / dx;
   dyi   = (real)1.0 / dy;
   dzi   = (real)1.0 / dz;

   dxyi  = dxi*dyi;
   dxzi  = dxi*dzi;
   dyzi  = dyi*dzi;
   dxyzi = dxi*dyi*dzi;


// determine location in table
   ix = 1 + (int)( ( x - xt[0] )*dxi );
   iy = 1 + (int)( ( y - yt[0] )*dyi );
   iz = 1 + (int)( ( z - zt[0] )*dzi );

   ix = MAX(  1, MIN( ix, nx-1 )  );
   iy = MAX(  1, MIN( iy, ny-1 )  );
   iz = MAX(  1, MIN( iz, nz-1 )  );


// set up aux vars for interpolation
   delx = xt[ix] - x;
   dely = yt[iy] - y;
   delz = zt[iz] - z;


   idx = ix + nx*( iy + ny*iz );

   for (int i=0; i<nvars; i++)
   {
      iv = nxyz*TargetIdx[i] + idx;

//    set up aux vars for interpolation assuming array ordering (iv, ix, iy, iz)
      fh[0] = alltables[ iv                ]; // ( ix,   iy,   iz   )
      fh[1] = alltables[ iv - 1            ]; // ( ix-1, iy,   iz   )
      fh[2] = alltables[ iv     - nx       ]; // ( ix,   iy-1, iz   )
      fh[3] = alltables[ iv          - nxy ]; // ( ix,   iy,   iz-1 )
      fh[4] = alltables[ iv - 1 - nx       ]; // ( ix-1, iy-1, iz   )
      fh[5] = alltables[ iv - 1      - nxy ]; // ( ix-1, iy,   iz-1 )
      fh[6] = alltables[ iv     - nx - nxy ]; // ( ix,   iy-1, iz-1 )
      fh[7] = alltables[ iv - 1 - nx - nxy ]; // ( ix-1, iy-1, iz-1 )

//    set up coeffs of interpolation polynomical and evaluate function values
      a[0] = fh[0];
      a[1] = dxi  *( fh[1] - fh[0] );
      a[2] = dyi  *( fh[2] - fh[0] );
      a[3] = dzi  *( fh[3] - fh[0] );
      a[4] = dxyi *( fh[4] - fh[1] - fh[2] + fh[0] );
      a[5] = dxzi *( fh[5] - fh[1] - fh[3] + fh[0] );
      a[6] = dyzi *( fh[6] - fh[2] - fh[3] + fh[0] );
      a[7] = dxyzi*( fh[7] - fh[0] + fh[1] + fh[2] +
                     fh[3] - fh[4] - fh[5] - fh[6] );

      output_vars[i] = a[0]
                     + a[1]*delx
                     + a[2]*dely
                     + a[3]*delz
                     + a[4]*delx*dely
                     + a[5]*delx*delz
                     + a[6]*dely*delz
                     + a[7]*delx*dely*delz;
   } // for (int i=0; i<nvars; i++)

} // FUNCTION : nuc_eos_C_linterp_some



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
