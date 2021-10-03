#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

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

#endif // #ifdef __CUDACC__ ... else ...




//-------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_cubinterp_some
// Description :  Find thermodynamic variables using 3D Catmull-Rom cubic interpolation
//                formula by searching the tabulated nuclear EoS
//
// Note        :  1. Invoked by nuc_eos_C_short()
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
void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               const int *TargetIdx, real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt )
{

   const int   nxy  = nx*ny;
   const int   nxyz = nxy*nz;
   const real *pv   = NULL;
         real  r, q, vox;
         real  u[4], v[4], w[4];


// determine spacing parameters of equidistant (!!!) table
   const real dx  = ( xt[nx-1] - xt[0] ) / (real)(nx-1);
   const real dy  = ( yt[ny-1] - yt[0] ) / (real)(ny-1);
   const real dz  = ( zt[nz-1] - zt[0] ) / (real)(nz-1);

   const real dxi = (real)1.0 / dx;
   const real dyi = (real)1.0 / dy;
   const real dzi = (real)1.0 / dz;


// determine location in table
   const int ix = (int)( ( x - xt[0] )*dxi );
   const int iy = (int)( ( y - yt[0] )*dyi );
   const int iz = (int)( ( z - zt[0] )*dzi );


// linear interpolation at boundaries
   if ( ix ==    0  ||  iy ==    0  ||  iz ==    0  ||
        ix >= nx-2  ||  iy >= ny-2  ||  iz >= nz-2     )
   {
      nuc_eos_C_linterp_some( x, y, z, TargetIdx, output_vars, alltables,
                              nx, ny, nz, nvars, xt, yt, zt );
      return;
   }


// difference
   const real delx = ( x - xt[ix] )*dxi;
   const real dely = ( y - yt[iy] )*dyi;
   const real delz = ( z - zt[iz] )*dzi;


// factors for Catmull-Rom interpolation
   const real delx2 =  SQR( delx );
   const real dely2 =  SQR( dely );
   const real delz2 =  SQR( delz );
   const real delx3 = CUBE( delx );
   const real dely3 = CUBE( dely );
   const real delz3 = CUBE( delz );

   u[0] = (real)-0.5*delx3 +           delx2 - (real)0.5*delx;
   u[1] = (real) 1.5*delx3 - (real)2.5*delx2 + (real)1.0;
   u[2] = (real)-1.5*delx3 + (real)2.0*delx2 + (real)0.5*delx;
   u[3] = (real) 0.5*delx3 - (real)0.5*delx2;

   v[0] = (real)-0.5*dely3 +           dely2 - (real)0.5*dely;
   v[1] = (real) 1.5*dely3 - (real)2.5*dely2 + (real)1.0;
   v[2] = (real)-1.5*dely3 + (real)2.0*dely2 + (real)0.5*dely;
   v[3] = (real) 0.5*dely3 - (real)0.5*dely2;

   w[0] = (real)-0.5*delz3 +           delz2 - (real)0.5*delz;
   w[1] = (real) 1.5*delz3 - (real)2.5*delz2 + (real)1.0;
   w[2] = (real)-1.5*delz3 + (real)2.0*delz2 + (real)0.5*delz;
   w[3] = (real) 0.5*delz3 - (real)0.5*delz2;


   const int iv = (ix-1) + (iy-1)*nx + (iz-1)*nxy;

   for (int idx=0; idx<nvars; idx++)
   {
      pv  = alltables + iv + nxyz*TargetIdx[idx];
      vox = (real)0.0;

      for (int k=0; k<4; k++)
      {
         q = (real)0.0;

         for (int j=0; j<4; j++)
         {
            r = (real)0.0;

            for (int i=0; i<4; i++)
            {
               r  += (*pv)*u[i];
               pv += 1;
            }

            q  += r*v[j];
            pv += nx - 4;
         }

         vox += q*w[k];
         pv  += nxy - 4*nx;
      } // for (int k=0; k<4; k++)

      output_vars[idx] = vox;
   } // for (int idx=0; idx<nvars; idx++)


// try linear interpolation if cubic interpolation fails
   if ( output_vars[0] != output_vars[0] )
      nuc_eos_C_linterp_some( x, y, z, TargetIdx, output_vars, alltables,
                              nx, ny, nz, nvars, xt, yt, zt );

} // FUNCTION : nuc_eos_C_cubinterp_some



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
