#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR  &&  NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )




//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_Func_CCSN
// Description :  Function to reset the temperature initial guess TEMP_IG in the CCSN problem
//
// Note        :  1. Invoked by Flu_ResetByUser_API_CCSN() and Model_Init_ByFunction_AssignData() using the
//                   function pointer "Flu_ResetByUser_Func_Ptr", which must be set by a test problem initializer
//                2. This function will be invoked when constructing the initial condition in
//                   Model_Init_ByFunction_AssignData() and after each update in Flu_ResetByUser_API_CCSN()
//                3. Input fluid[] stores the original values
//                4. Even when DUAL_ENERGY is adopted, one does NOT need to set the dual-energy variable here
//                   --> It will be set automatically in Flu_ResetByUser_API_CCSN() and
//                       Model_Init_ByFunction_AssignData()
//                5. Enabled by the runtime option "OPT__RESET_FLUID"
//
// Parameter   :  fluid    : Fluid array storing both the input (original) and reset values
//                           --> Including both active and passive variables
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                dt       : Time interval to advance solution
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  true  : This cell has been reset
//                false : This cell has not been reset
//-------------------------------------------------------------------------------------------------------
bool Flu_ResetByUser_Func_CCSN( real fluid[], const double x, const double y, const double z, const double Time,
                                const double dt, const int lv, double AuxArray[] )
{

   if ( AuxArray != NULL ) {
      const double Emag = AuxArray[0];
      const bool   CheckMinTemp_No = false;

      fluid[TEMP_IG] = Hydro_Con2Temp( fluid[DENS], fluid[MOMX], fluid[MOMY], fluid[MOMZ], fluid[ENGY],
                                       fluid+NCOMP_FLUID, CheckMinTemp_No, NULL_REAL, Emag,
                                       EoS_DensEint2Temp_CPUPtr, EoS_AuxArray_Flt, EoS_AuxArray_Int, h_EoS_Table );

      return true;
   }

   else
      return false;

} // FUNCTION : Flu_ResetByUser_Func_CCSN



//-------------------------------------------------------------------------------------------------------
// Function    :  Flu_ResetByUser_API_CCSN
// Description :  API for resetting the fluid array in the CCSN problem
//
// Note        :  1. Enabled by the runtime option "OPT__RESET_FLUID"
//                2. Invoked by EvolveLevel() using the function pointer "Flu_ResetByUser_API_Ptr"
//                   --> This function pointer is resey by Init_TestProb_Hydro_CCSN()
//                3. Currently NOT applied to the input uniform array
//                   --> Init_ByFile() does NOT call this function
//                4. Currently does not work with "OPT__OVERLAP_MPI"
//
// Parameter   :  lv      : Target refinement level
//                FluSg   : Target fluid sandglass
//                TimeNew : Current physical time (system has been updated from TimeOld to TimeNew in EvolveLevel())
//                dt      : Time interval to advance solution (can be different from TimeNew-TimeOld in COMOVING)
//-------------------------------------------------------------------------------------------------------
void Flu_ResetByUser_API_CCSN( const int lv, const int FluSg, const double TimeNew, const double dt )
{

   real   fluid[NCOMP_TOTAL];
   double AuxArray[1];
   bool   Reset;

#  pragma omp parallel for private( fluid, Reset ) schedule( runtime )
   for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
   {
      for (int k=0; k<PS1; k++)  {
      for (int j=0; j<PS1; j++)  {
      for (int i=0; i<PS1; i++)  {

         for (int v=0; v<NCOMP_TOTAL; v++)   fluid[v] = amr->patch[FluSg][lv][PID]->fluid[v][k][j][i];

#        ifdef MHD
         const real Emag = MHD_GetCellCenteredBEnergyInPatch( lv, PID, i, j, k, amr->MagSg[lv] );
#        else
         const real Emag = NULL_REAL;
#        endif
         AuxArray[1] = Emag;

//       reset temperature initial guess for this cell
         Reset = Flu_ResetByUser_Func_CCSN( fluid, NULL_REAL, NULL_REAL, NULL_REAL, TimeNew, dt, lv, AuxArray );

//       store the reset values
         if ( Reset )
             amr->patch[FluSg][lv][PID]->fluid[TEMP_IG][k][j][i] = fluid[TEMP_IG];

      }}} // i,j,k
   } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)

} // FUNCTION : Flu_ResetByUser_API_CCSN



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR  &&  NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
