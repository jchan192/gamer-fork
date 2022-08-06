#include "GAMER.h"
#include "NuclearEoS.h"


extern double CCSN_LB_TimeFac;



//-------------------------------------------------------------------------------------------------------
// Function    :  Mis_GetTimeStep_Lightbulb
// Description :  Estimate the evolution time-step constrained by the lightbulb source term
//
// Note        :  1. This function should be applied to both physical and comoving coordinates and always
//                   return the evolution time-step (dt) actually used in various solvers
//                   --> Physical coordinates : dt = physical time interval
//                       Comoving coordinates : dt = delta(scale_factor) / ( Hubble_parameter*scale_factor^3 )
//                   --> We convert dt back to the physical time interval, which equals "delta(scale_factor)"
//                       in the comoving coordinates, in Mis_GetTimeStep()
//                2. Invoked by Mis_GetTimeStep() using the function pointer "Mis_GetTimeStep_User_Ptr",
//                   which must be set by a test problem initializer
//                3. Enabled by the runtime option "OPT__DT_USER"
//
// Parameter   :  lv       : Target refinement level
//                dTime_dt : dTime/dt (== 1.0 if COMOVING is off)
//
// Return      :  dt
//-------------------------------------------------------------------------------------------------------
double Mis_GetTimeStep_Lightbulb( const int lv, const double dTime_dt )
{

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   double   dt_LB         = HUGE_NUMBER;
   double   dt_LB_Inv     = -__DBL_MAX__;
   double **OMP_dt_LB_Inv = NULL;
   Aux_AllocateArray2D( OMP_dt_LB_Inv, NT, 1 );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_dt_LB_Inv[TID][0] = -__DBL_MAX__;

      const double dh = amr->dh[lv];

#     pragma omp for schedule( runtime )
      for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      {
         for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
         for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
         for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

            const real Dens       = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
            const real Momx       = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMX][k][j][i];
            const real Momy       = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMY][k][j][i];
            const real Momz       = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMZ][k][j][i];
            const real Engy       = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[ENGY][k][j][i];
#           ifdef DELE
                  real dEint_Code = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DELE][k][j][i];
#           else
                  real dEint_Code = NULL_REAL;
#           endif

#           ifdef MHD
            const real Emag = MHD_GetCellCenteredBEnergy( amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGX],
                                                          amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGY],
                                                          amr->patch[ amr->MagSg[lv] ][lv][PID]->magnetic[MAGZ],
                                                          PS1, PS1, PS1, i, j, k );
#           else
            const real Emag = NULL_REAL;
#           endif

            const real Eint_Code = Hydro_Con2Eint( Dens, Momx, Momy, Momz, Engy, true, MIN_EINT, Emag );


//          call Src_Lightbulb() to get the heating/cooling rate if not computed yet
            if ( dEint_Code == NULL_REAL )
            {
//             get the input arrays
               real fluid[FLU_NIN_S];

               for (int v=0; v<FLU_NIN_S; v++)  fluid[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#              ifdef MHD
               real B[NCOMP_MAG] = { Emag, 0.0, 0.0 };
#              else
               real *B = NULL;
#              endif


               SrcTerms.Lightbulb_FuncPtr( fluid, B, &SrcTerms, NULL_REAL, dh, x, y, z, NULL_REAL, NULL_REAL,
                                           MIN_DENS, MIN_PRES, MIN_EINT, &EoS,
                                           SrcTerms.Lightbulb_AuxArrayDevPtr_Flt, SrcTerms.Lightbulb_AuxArrayDevPtr_Int );

#              ifdef DELE
               dEint_Code = fluid[DELE];
#              endif
            } // if ( dEint_Code == NULL_REAL )


            const double _Eint_Ratio = fabs( dEint_Code / Eint_Code );

//          compare the inverse of ratio to avoid zero division, and store the maximum value
            if ( _Eint_Ratio > OMP_dt_LB_Inv[TID][0] )   OMP_dt_LB_Inv[TID][0] = _Eint_Ratio;

         }}} // i,j,k
      } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
   {
      if ( OMP_dt_LB_Inv[TID][0] > dt_LB_Inv )   dt_LB_Inv = OMP_dt_LB_Inv[TID][0];
   }

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_dt_LB_Inv );


// find the maximum over all MPI processes
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &dt_LB_Inv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
#  endif


   dt_LB = CCSN_LB_TimeFac / dt_LB_Inv;

   return dt_LB;

} // FUNCTION : Mis_GetTimeStep_Lightbulb
