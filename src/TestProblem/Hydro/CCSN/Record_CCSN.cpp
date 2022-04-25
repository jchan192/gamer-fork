#include "GAMER.h"


static double CCSN_CentralDens;

extern bool   CCSN_Is_PostBounce;



//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CCSN_CentralDens
// Description :  Record the maximum density
//
// Note        :  1. Invoked by Record_CCSN()
//-------------------------------------------------------------------------------------------------------
void Record_CCSN_CentralDens()
{

   const char   filename_central_dens[] = "Record__CentralDens";
   const double BoxCenter[3]            = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   double   DataCoord[4]  = { -__DBL_MAX__ };
   double **OMP_DataCoord = NULL;
   Aux_AllocateArray2D( OMP_DataCoord, NT, 4 );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_DataCoord[TID][0] = -__DBL_MAX__;
      for (int b=1; b<4; b++)   OMP_DataCoord[TID][b] = 0.0;

      for (int lv=0; lv<NLEVEL; lv++)
      {
         const double dh = amr->dh[lv];

#        pragma omp for schedule( runtime )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

               const double dens = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];

               if ( dens > OMP_DataCoord[TID][0] )
               {
                  OMP_DataCoord[TID][0] = dens;
                  OMP_DataCoord[TID][1] = x;
                  OMP_DataCoord[TID][2] = y;
                  OMP_DataCoord[TID][3] = z;
               }

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<NLEVEL; lv++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
   {
      if ( OMP_DataCoord[TID][0] > DataCoord[0] )
         for (int b=0; b<4; b++)   DataCoord[b] = OMP_DataCoord[TID][b];
   }

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_DataCoord );


// collect data from all ranks
#  ifndef SERIAL
   {
      double DataCoord_All[4 * MPI_NRank];

      MPI_Allgather( DataCoord, 4, MPI_DOUBLE, DataCoord_All, 4, MPI_DOUBLE, MPI_COMM_WORLD );

      for (int i=0; i<MPI_NRank; i++)
      {
         if ( DataCoord_All[4 * i] > DataCoord[0] )
            for (int b=0; b<4; b++)   DataCoord[b] = DataCoord_All[4 * i + b];
      }
   }
#  endif // ifndef SERIAL


// write to the file "Record__CentralDens"
   if ( MPI_Rank == 0 )
   {

      static bool FirstTime = true;

//    file header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(filename_central_dens) )
         {
             Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", filename_central_dens );
         }

         else
         {
             FILE *file_max_dens = fopen( filename_central_dens, "w" );
             fprintf( file_max_dens, "#%14s %12s %16s %16s %16s %16s\n",
                                     "Time [sec]", "Step", "Dens [g/cm^3]", "PosX [cm]", "PosY [cm]", "PosZ [cm]" );
             fclose( file_max_dens );
         }

         FirstTime = false;
      }

      FILE *file_max_dens = fopen( filename_central_dens, "a" );
      fprintf( file_max_dens, "%15.7e %12ld %16.7e %16.7e %16.7e %16.7e\n",
               Time[0]*UNIT_T, Step, DataCoord[0]*UNIT_D, DataCoord[1]*UNIT_L, DataCoord[2]*UNIT_L, DataCoord[3]*UNIT_L );
      fclose( file_max_dens );

   } // if ( MPI_Rank == 0 )


// store the central density in cgs unit for detecting core bounces
   CCSN_CentralDens = DataCoord[0] * UNIT_D;

} // FUNCTION : Record_CCSN_CentralDens()



#ifdef GRAVITY
//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CCSN_GWSignal
// Description :  Record the second-order time derivative of mass quadrupole moments
//
// Note        :  1. Invoked by Record_CCSN()
//                2. Ref: Kenichi Oohara, et al., 1997, PThPS, 128, 183 (arXiv: 1206.4724), sec. 2.1
//-------------------------------------------------------------------------------------------------------
void Record_CCSN_GWSignal()
{

   const char   filename_QuadMom_2nd[ ] = "Record__QuadMom_2nd";
   const double BoxCenter           [3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   const int NData   = 6;
   const int ArrayID = 0;
   const int NPG_Max = POT_GPU_NPGROUP;

   double   QuadMom_2nd[NData] = { 0.0 };
   double **OMP_QuadMom_2nd    = NULL;
   Aux_AllocateArray2D( OMP_QuadMom_2nd, NT, NData );

   for (int TID=0; TID<NT; TID++) {
   for (int b=0; b<NData; b++)    {
      OMP_QuadMom_2nd[TID][b] = 0.0;
   }}


   for (int lv=0; lv<NLEVEL; lv++)
   {
      const double dh        = amr->dh[lv];
      const double dv        = CUBE( dh );
      const double TimeNew   = Time[lv];
      const int    NTotal    = amr->NPatchComma[lv][1] / 8;
            int   *PID0_List = new int [NTotal];

      for (int t=0; t<NTotal; t++)  PID0_List[t] = 8*t;

      for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)
      {
//       prepare the potential file
         int NPG = ( NPG_Max < NTotal-Disp ) ? NPG_Max : NTotal-Disp;

         Prepare_PatchData( lv, TimeNew, &h_Pot_Array_P_Out[ArrayID][0][0][0][0], NULL,
                            GRA_GHOST_SIZE, NPG, PID0_List+Disp, _POTE, _NONE,
                            OPT__GRA_INT_SCHEME, INT_NONE, UNIT_PATCH, (GRA_GHOST_SIZE==0)?NSIDE_00:NSIDE_06, false,
                            OPT__BC_FLU, OPT__BC_POT, -1.0, -1.0, -1.0, -1.0, false );

#        pragma omp parallel for schedule( runtime )
         for (int PID_IDX=0; PID_IDX<8*NPG; PID_IDX++)
         {
#           ifdef OPENMP
            const int TID = omp_get_thread_num();
#           else
            const int TID = 0;
#           endif

            const int PID = 8*Disp + PID_IDX;

            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh; const int kk = k + GRA_GHOST_SIZE;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh; const int jj = j + GRA_GHOST_SIZE;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh; const int ii = i + GRA_GHOST_SIZE;

               const double dx = x - BoxCenter[0];
               const double dy = y - BoxCenter[1];
               const double dz = z - BoxCenter[2];
               const double r = sqrt( SQR(dx) + SQR(dy) + SQR(dz) );

               const double dens  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];
               const double momx  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMX][k][j][i];
               const double momy  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMY][k][j][i];
               const double momz  = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[MOMZ][k][j][i];
               const double _dens = 1.0 / dens;

               const real (*PrepPotPtr)[GRA_NXT][GRA_NXT] = h_Pot_Array_P_Out[ArrayID][PID_IDX];

//             compute the potential gradient through central difference method
               const double dPhi_dx = ( PrepPotPtr[kk  ][jj  ][ii+1] - PrepPotPtr[kk  ][jj  ][ii-1] ) / (2.0 * dh);
               const double dPhi_dy = ( PrepPotPtr[kk  ][jj+1][ii  ] - PrepPotPtr[kk  ][jj-1][ii  ] ) / (2.0 * dh);
               const double dPhi_dz = ( PrepPotPtr[kk+1][jj  ][ii  ] - PrepPotPtr[kk-1][jj  ][ii  ] ) / (2.0 * dh);

               const double trace = _dens * ( SQR(momx) + SQR(momy) + SQR(momz) )
                                  -  dens * ( dx * dPhi_dx + dy * dPhi_dy + dz * dPhi_dz );

               OMP_QuadMom_2nd[TID][0] += dv * ( 2.0 * _dens * momx * momx - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dx * dPhi_dx                      );  // Ixx
               OMP_QuadMom_2nd[TID][1] += dv * ( 2.0 * _dens * momx * momy
                                               -        dens * ( dx * dPhi_dy + dy * dPhi_dx )   );  // Ixy
               OMP_QuadMom_2nd[TID][2] += dv * ( 2.0 * _dens * momx * momz
                                               -        dens * ( dx * dPhi_dz + dz * dPhi_dx )   );  // Ixz
               OMP_QuadMom_2nd[TID][3] += dv * ( 2.0 * _dens * momy * momy - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dy * dPhi_dy                      );  // Iyy
               OMP_QuadMom_2nd[TID][4] += dv * ( 2.0 * _dens * momy * momz
                                               -        dens * ( dy * dPhi_dz + dz * dPhi_dy )   );  // Iyz
               OMP_QuadMom_2nd[TID][5] += dv * ( 2.0 * _dens * momz * momz - (2.0 / 3.0) * trace
                                               - 2.0 *  dens * dz * dPhi_dz                      );  // Izz

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int Disp=0; Disp<NTotal; Disp+=NPG_Max)

      delete [] PID0_List;
   } // for (int lv=0; lv<NLEVEL; lv++)


// sum over all OpenMP threads
   for (int b=0; b<NData; b++) {
   for (int t=0; t<NT; t++)    {
      QuadMom_2nd[b] += OMP_QuadMom_2nd[t][b];
   }}

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_QuadMom_2nd );


// collect data from all ranks (in-place reduction)
#  ifndef SERIAL
   if ( MPI_Rank == 0 )   MPI_Reduce( MPI_IN_PLACE, QuadMom_2nd, NData, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
   else                   MPI_Reduce( QuadMom_2nd,  NULL,        NData, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


// multiply the coefficient (G / c^4) and unit
   const double QuadMom_fac = UNIT_M * SQR( UNIT_V ) * Const_NewtonG / pow( Const_c, 4.0 );

   for (int b=0; b<NData; b++)   QuadMom_2nd[b] *= QuadMom_fac;


// write to the file "Record__QuadMom_2nd"
   if ( MPI_Rank == 0 )
   {

      static bool FirstTime = true;

//    file header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(filename_QuadMom_2nd) )
         {
             Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", filename_QuadMom_2nd );
         }
         else
         {
             FILE *file_QuadMom_2nd = fopen( filename_QuadMom_2nd, "w" );
             fprintf( file_QuadMom_2nd, "#%14s %12s %16s %16s %16s %16s %16s %16s\n",
                                        "Time [sec]", "Step", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz" );
             fclose( file_QuadMom_2nd );
         }

         FirstTime = false;
      }

      FILE *file_QuadMom_2nd = fopen( filename_QuadMom_2nd, "a" );

                                    fprintf( file_QuadMom_2nd, "%15.7e %12ld", Time[0] * UNIT_T, Step );
      for (int b=0; b<NData; b++)   fprintf( file_QuadMom_2nd, "%17.7e", QuadMom_2nd[b] );
                                    fprintf( file_QuadMom_2nd, "\n" );

      fclose( file_QuadMom_2nd );

   } // if ( MPI_Rank == 0 )

} // FUNCTION : Record_CCSN_GWSignal()
#endif // ifdef GRAVITY



//-------------------------------------------------------------------------------------------------------
// Function    :  Detect_CoreBounce
// Description :  Check whether the core bounce occurs
//
// Note        :  1. Invoked by Record_CCSN()
//             :  2. Based on two criteria:
//                   --> (a) The central density is larger than 2e14
//                       (b) Any cells within 30km has entropy larger than 3
//-------------------------------------------------------------------------------------------------------
void Detect_CoreBounce()
{

// (1) criterion 1: central density is larger than 2e14
   if ( CCSN_CentralDens < 2e14 )   return;


// (2) criterion 2: any cells within 30km has entropy larger than 3
   const double BoxCenter[3] = { amr->BoxCenter[0], amr->BoxCenter[1], amr->BoxCenter[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;
#  else
   const int NT = 1;
#  endif

   double MaxEntr = -__DBL_MAX__;
   double OMP_MaxEntr[NT];


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_MaxEntr[TID] = -__DBL_MAX__;

      for (int lv=0; lv<NLEVEL; lv++)
      {
         const double dh = amr->dh[lv];

#        pragma omp for schedule( runtime )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

               const double x0 = x - BoxCenter[0];
               const double y0 = y - BoxCenter[1];
               const double z0 = z - BoxCenter[2];
               const double r  = sqrt(  SQR( x0 ) + SQR( y0 ) + SQR( z0 )  );

//             ignore cells outside 30km
               if ( r * UNIT_L > 3.0e6 )   continue;

//             retrieve the entropy and store the maximum value
               real u[NCOMP_TOTAL], Entr, Emag=NULL_REAL;

               for (int v=0; v<NCOMP_TOTAL; v++)   u[v] = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[v][k][j][i];

#              ifdef MHD
               Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#              endif

               Entr = Hydro_Con2Entr( u[DENS], u[MOMX], u[MOMY], u[MOMZ], u[ENGY], u+NCOMP_FLUID,
                                      false, NULL_REAL, Emag, EoS_DensEint2Entr_CPUPtr,
                                      EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

               OMP_MaxEntr[TID] = MAX( OMP_MaxEntr[TID], (double)Entr );

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<NLEVEL; lv++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
      MaxEntr = MAX( MaxEntr, OMP_MaxEntr[TID] );


// collect data from all ranks
#  ifndef SERIAL
   MPI_Allreduce( MPI_IN_PLACE, &MaxEntr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
#  endif // ifndef SERIAL


   if ( MaxEntr > 3.0 )   CCSN_Is_PostBounce = true;

} // FUNCTION : Detect_CoreBounce()
