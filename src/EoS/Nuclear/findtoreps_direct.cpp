#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



//-------------------------------------------------------------------------------------
// Function    :  BinarySearch
// Description :  Use the binary method to find the index
//
// Note        :  1. Invoked by findtoreps_direct()
//
// Return      :  Mid
//-------------------------------------------------------------------------------------
GPU_DEVICE
int BinarySearch( const real *table, const real target, int Min, int Max,
                  const int nx, const int ip )
{

   int Mid = -1;

   while (  ( Mid = ( Min + Max ) / 2 ) != Min  )
   {
      if   ( table[ip + nx * Mid] > target )   Max = Mid;
      else                                     Min = Mid;
   }

   return Mid;

} // FUNCTION : BinarySearch



//-------------------------------------------------------------------------------------
// Function    :  Bilinear
// Description :  Bilinear interpolation
//
// Note        :  1. Invoked by findtoreps_direct()
//
// Return      :  interpolated value
//-------------------------------------------------------------------------------------
GPU_DEVICE
real Bilinear( const real *table, const real frac_x, const real frac_y,
               const int ifs, const int nxy )
{

   const real left  = table[ifs      ] + ( table[ifs + 1      ] - table[ifs      ] ) * frac_x;
   const real right = table[ifs + nxy] + ( table[ifs + 1 + nxy] - table[ifs + nxy] ) * frac_x;

   return left + ( right - left ) * frac_y;

} // FUNCTION : Bilinear



//-------------------------------------------------------------------------------------
// Function    :  findtoreps_direct
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
//                var           : Input vector of second variable (eps/temp/entr/pres)
//                z             : Input vector of third  variable (Ye)
//                found_ltoreps : Output log(temp)/log(eps) of interpolated function values
//                table_main    : The main      tabulated nuclear EoS table
//                table_Aux     : The auxiliary tabulated nuclear EoS table
//                nx            : X-dimension of table
//                ny            : Y-dimension of table
//                nz            : Z-dimension of table
//                nvar          : Size of input quantity in the auxiliary table
//                xt            : Vector of x-coordinates of table
//                yt            : Vector of y-coordinates of table
//                zt            : Vector of z-coordinates of table
//                vart          : Vector of the input quantity
//                keymode       : Which mode to be used
//                keyerr        : Output error
//
// Return      :  found_ltoreps
//-------------------------------------------------------------------------------------
GPU_DEVICE
void findtoreps_direct( const real x, const real var, const real z,
                        real *found_ltoreps, const real *table_main, const real *table_aux,
                        const int nx, const int ny, const int nz, const int nvar,
                        const real *xt, const real *yt, const real *zt, const real *vart,
                        const int keymode, int *keyerr )
{

   int  ix, iy_brd1, iy_brd2, iz, iv, ip, ifs;
   real dx, dz, dv, dxi, dzi, dvi, fracx, fracy, fracz;
   real var_brd1, var_brd2;

   const int nxy  = nx  * ny;
   const int nxyz = nxy * nz;


// determine spacing parameters and location in an equidistant table
   dx = (   xt[nx  -1] -   xt[0] ) / (real)(nx  -1);  // dens
   dz = (   zt[nz  -1] -   zt[0] ) / (real)(nz  -1);  // ye
   dv = ( vart[nvar-1] - vart[0] ) / (real)(nvar-1);  // var

   dxi = (real)1.0 / dx;
   dzi = (real)1.0 / dz;
   dvi = (real)1.0 / dv;

   ix = (int)(  (   x -   xt[0] ) * dxi  );
   iz = (int)(  (   z -   zt[0] ) * dzi  );
   iv = (int)(  ( var - vart[0] ) * dvi  );  // index in the auxiliary table


// (0) check if the target variable is in the degeneracy region
   const int iv_left  =     (int)table_aux[ix + nx *  iv    + nxy * iz];  // index in the main table
   const int iv_right = 1 + (int)table_aux[ix + nx * (iv+1) + nxy * iz];  // index in the main table

   if (  ( iv_left < 0 )  ||  ( iv_right < 0 )  )   {  *keyerr = 660;  return;  }

// compute the fraction and the corresponding location in the main table
   if      ( keymode == NUC_MODE_ENGY )   ip = NUC_VAR_IDX_EORT;  // energy mode
   else if ( keymode == NUC_MODE_ENTR )   ip = NUC_VAR_IDX_ENTR;  // entropy mode
   else if ( keymode == NUC_MODE_PRES )   ip = NUC_VAR_IDX_PRES;  // pressure mode

   ip = ip * nxyz + ix + nxy * iz;

   fracx = ( x - xt[ix] ) * dxi;
   fracz = ( z - zt[iz] ) * dzi;


// (1-a) find the index `iy_brd1` s.t. var(ix, iy_brd1, iz) <= var < var(ix, iy_brd1+1, iz)
   iy_brd1 = BinarySearch( table_main, var, iv_left, iv_right, nx, ip );

   if ( iy_brd1 < 0 )   {  *keyerr = 660;  return;  }

// (1-b) find the value at the given density and Ye, and the input quantity at `iy_brd1`
   ifs      = ip + nx * iy_brd1;
   var_brd1 = Bilinear( table_main, fracx, fracz, ifs, nxy );


// (2) find the lower and upper boundaries iteratively
   bool brd_found = false;

   if ( var_brd1 > var )
   {
//    (2-a) guess the solution located at index < iy_brd1, and find the lower boundary
//      while ( iy_brd1 > iv_left ) // TO CHECK: unstable, check again after the PR for IG of temperature
      while ( iy_brd1 > 0 )
      {
         iy_brd2  = iy_brd1 - 1;
         ifs      = ip + nx * iy_brd2;
         var_brd2 = Bilinear( table_main, fracx, fracz, ifs, nxy );

         if   ( var_brd2 < var )   {  brd_found = true;  break;  }
         else                      {  iy_brd1 = iy_brd2;  var_brd1 = var_brd2;  }
      }
   }

   else
   {
//    (2-b) guess the solution located at index > iy_brd1, and find the upper boundary
//      while ( iy_brd1 < iv_right ) // TO CHECK: unstable, check again after the PR for IG of temperature
      while ( iy_brd1 < ny - 1 )
      {
         iy_brd2  = iy_brd1 + 1;
         ifs      = ip + nx * iy_brd2;
         var_brd2 = Bilinear( table_main, fracx, fracz, ifs, nxy );

         if   ( var_brd2 > var )   {  brd_found = true;  break;  }
         else                      {  iy_brd1 = iy_brd2;  var_brd1 = var_brd2;  }
      }
   } // if ( var_brd1 > var ) ... else ...


// (3) apply linear interpolation to compute the temperature / energy, if found
   if ( brd_found == false )   {  *keyerr = 660;  return;  }

   fracy = ( var - var_brd1 ) / ( var_brd2 - var_brd1 );

   *found_ltoreps = yt[iy_brd1] + ( yt[iy_brd2] - yt[iy_brd1] ) * fracy;

   return;

} // FUNCTION : findtoreps_direct



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
