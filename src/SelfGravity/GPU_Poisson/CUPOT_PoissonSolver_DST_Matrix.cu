#include "Macro.h"
#include "CUPOT.h"

#if ( defined GRAVITY  &&  defined GPU  &&  POT_SCHEME == DST )


#define POT_NXT_F    ( PATCH_SIZE+2*POT_GHOST_SIZE           )
#define POT_PAD      ( WARP_SIZE/2 - (POT_NXT_F*2%WARP_SIZE) )
#define POT_NTHREAD  ( RHO_NXT*RHO_NXT/2    )
#define POT_USELESS  ( POT_GHOST_SIZE%2                      )

#if (RHO_NXT==24)
  __device__ real M[24*24] = {0.12533323356430426 , 0.2486898871648548 , 0.3681245526846779 , 0.4817536741017153 , 0.5877852522924731 , 0.6845471059286886 , 0.7705132427757891 , 0.8443279255020151 , 0.9048270524660196 , 0.9510565162951535 , 0.9822872507286886 , 0.9980267284282716 , 0.9980267284282716 , 0.9822872507286887 , 0.9510565162951536 , 0.9048270524660195 , 0.844327925502015 , 0.7705132427757893 , 0.6845471059286888 , 0.5877852522924732 , 0.4817536741017152 , 0.36812455268467814 , 0.24868988716485524 , 0.12533323356430454 , 0.2486898871648548 , 0.4817536741017153 , 0.6845471059286886 , 0.8443279255020151 , 0.9510565162951535 , 0.9980267284282716 , 0.9822872507286887 , 0.9048270524660195 , 0.7705132427757893 , 0.5877852522924732 , 0.36812455268467814 , 0.12533323356430454 , -0.1253332335643043 , -0.3681245526846779 , -0.5877852522924727 , -0.7705132427757894 , -0.9048270524660198 , -0.9822872507286887 , -0.9980267284282716 , -0.9510565162951536 , -0.8443279255020151 , -0.684547105928689 , -0.4817536741017161 , -0.24868988716485535 , 0.3681245526846779 , 0.6845471059286886 , 0.9048270524660196 , 0.9980267284282716 , 0.9510565162951536 , 0.7705132427757893 , 0.4817536741017152 , 0.12533323356430454 , -0.24868988716485457 , -0.5877852522924727 , -0.8443279255020147 , -0.9822872507286887 , -0.9822872507286886 , -0.8443279255020151 , -0.587785252292474 , -0.24868988716485535 , 0.12533323356430506 , 0.48175367410171493 , 0.7705132427757887 , 0.9510565162951532 , 0.9980267284282716 , 0.90482705246602 , 0.6845471059286884 , 0.368124552684678 , 0.4817536741017153 , 0.8443279255020151 , 0.9980267284282716 , 0.9048270524660195 , 0.5877852522924732 , 0.12533323356430454 , -0.3681245526846779 , -0.7705132427757894 , -0.9822872507286887 , -0.9510565162951536 , -0.684547105928689 , -0.24868988716485535 , 0.2486898871648549 , 0.6845471059286886 , 0.9510565162951532 , 0.9822872507286886 , 0.7705132427757886 , 0.368124552684678 , -0.12533323356430318 , -0.5877852522924728 , -0.9048270524660197 , -0.9980267284282716 , -0.8443279255020161 , -0.48175367410171627 , 0.5877852522924731 , 0.9510565162951535 , 0.9510565162951536 , 0.5877852522924732 , 1.2246467991473532e-16 , -0.5877852522924727 , -0.9510565162951535 , -0.9510565162951536 , -0.5877852522924732 , -2.4492935982947064e-16 , 0.5877852522924722 , 0.9510565162951532 , 0.9510565162951536 , 0.5877852522924734 , 3.6739403974420594e-16 , -0.5877852522924728 , -0.9510565162951534 , -0.9510565162951538 , -0.5877852522924735 , -4.898587196589413e-16 , 0.5877852522924728 , 0.9510565162951529 , 0.9510565162951543 , 0.5877852522924751 , 0.6845471059286886 , 0.9980267284282716 , 0.7705132427757893 , 0.12533323356430454 , -0.5877852522924727 , -0.9822872507286887 , -0.8443279255020151 , -0.24868988716485535 , 0.48175367410171493 , 0.9510565162951532 , 0.90482705246602 , 0.368124552684678 , -0.3681245526846789 , -0.9048270524660197 , -0.9510565162951543 , -0.48175367410171627 , 0.24868988716485635 , 0.8443279255020146 , 0.9822872507286889 , 0.5877852522924751 , -0.1253332335643047 , -0.7705132427757879 , -0.9980267284282716 , -0.6845471059286887 , 0.7705132427757891 , 0.9822872507286887 , 0.4817536741017152 , -0.3681245526846779 , -0.9510565162951535 , -0.8443279255020151 , -0.12533323356430467 , 0.6845471059286886 , 0.9980267284282716 , 0.5877852522924734 , -0.24868988716485302 , -0.9048270524660197 , -0.9048270524660185 , -0.2486898871648556 , 0.5877852522924714 , 0.9980267284282716 , 0.6845471059286886 , -0.1253332335643047 , -0.8443279255020135 , -0.9510565162951538 , -0.36812455268467664 , 0.4817536741017121 , 0.9822872507286884 , 0.7705132427757888 , 0.8443279255020151 , 0.9048270524660195 , 0.12533323356430454 , -0.7705132427757894 , -0.9510565162951536 , -0.24868988716485535 , 0.6845471059286886 , 0.9822872507286886 , 0.368124552684678 , -0.5877852522924728 , -0.9980267284282716 , -0.48175367410171627 , 0.4817536741017155 , 0.9980267284282716 , 0.5877852522924751 , -0.3681245526846787 , -0.982287250728689 , -0.6845471059286887 , 0.2486898871648527 , 0.9510565162951533 , 0.7705132427757888 , -0.12533323356430268 , -0.9048270524660179 , -0.8443279255020163 , 0.9048270524660196 , 0.7705132427757893 , -0.24868988716485457 , -0.9822872507286887 , -0.5877852522924732 , 0.48175367410171493 , 0.9980267284282716 , 0.368124552684678 , -0.684547105928688 , -0.9510565162951538 , -0.12533323356430578 , 0.8443279255020146 , 0.8443279255020152 , -0.1253332335643047 , -0.9510565162951534 , -0.6845471059286887 , 0.3681245526846786 , 0.9980267284282714 , 0.48175367410171666 , -0.5877852522924724 , -0.9822872507286887 , -0.24868988716485782 , 0.7705132427757877 , 0.9048270524660202 , 0.9510565162951535 , 0.5877852522924732 , -0.5877852522924727 , -0.9510565162951536 , -2.4492935982947064e-16 , 0.9510565162951532 , 0.5877852522924734 , -0.5877852522924728 , -0.9510565162951538 , -4.898587196589413e-16 , 0.9510565162951529 , 0.5877852522924751 , -0.5877852522924726 , -0.9510565162951538 , -7.347880794884119e-16 , 0.9510565162951533 , 0.5877852522924738 , -0.5877852522924724 , -0.9510565162951539 , -9.797174393178826e-16 , 0.9510565162951532 , 0.5877852522924769 , -0.5877852522924694 , -0.951056516295155 , 0.9822872507286886 , 0.36812455268467814 , -0.8443279255020153 , -0.684547105928689 , 0.5877852522924729 , 0.9048270524660192 , -0.24868988716485474 , -0.9980267284282716 , -0.12533323356430404 , 0.9510565162951534 , 0.4817536741017165 , -0.7705132427757901 , -0.7705132427757888 , 0.4817536741017152 , 0.951056516295155 , -0.12533323356430268 , -0.9980267284282717 , -0.24868988716485438 , 0.9048270524660179 , 0.587785252292474 , -0.68454710592869 , -0.8443279255020164 , 0.3681245526846748 , 0.9822872507286882 , 0.9980267284282716 , 0.12533323356430454 , -0.9822872507286887 , -0.24868988716485535 , 0.9510565162951532 , 0.368124552684678 , -0.9048270524660197 , -0.48175367410171627 , 0.8443279255020146 , 0.5877852522924751 , -0.7705132427757879 , -0.6845471059286887 , 0.6845471059286902 , 0.7705132427757888 , -0.5877852522924695 , -0.8443279255020163 , 0.4817536741017181 , 0.9048270524660202 , -0.3681245526846749 , -0.951056516295155 , 0.24868988716485566 , 0.9822872507286895 , -0.1253332335643057 , -0.9980267284282716 , 0.9980267284282716 , -0.1253332335643043 , -0.9822872507286887 , 0.2486898871648549 , 0.9510565162951536 , -0.36812455268467725 , -0.90482705246602 , 0.4817536741017155 , 0.8443279255020152 , -0.5877852522924726 , -0.770513242775791 , 0.6845471059286876 , 0.6845471059286887 , -0.7705132427757878 , -0.5877852522924739 , 0.8443279255020153 , 0.48175367410171377 , -0.9048270524660194 , -0.3681245526846804 , 0.9510565162951532 , 0.24868988716485124 , -0.9822872507286876 , -0.125333233564305 , 0.9980267284282713 , 0.9822872507286887 , -0.3681245526846779 , -0.8443279255020151 , 0.6845471059286886 , 0.5877852522924734 , -0.9048270524660197 , -0.2486898871648556 , 0.9980267284282716 , -0.1253332335643047 , -0.9510565162951538 , 0.4817536741017121 , 0.7705132427757888 , -0.7705132427757924 , -0.48175367410171677 , 0.9510565162951522 , 0.12533323356430465 , -0.9980267284282716 , 0.24868988716485566 , 0.9048270524660219 , -0.587785252292472 , -0.6845471059286866 , 0.8443279255020112 , 0.36812455268468075 , -0.9822872507286889 , 0.9510565162951536 , -0.5877852522924727 , -0.5877852522924732 , 0.9510565162951532 , 3.6739403974420594e-16 , -0.9510565162951538 , 0.5877852522924728 , 0.5877852522924751 , -0.9510565162951534 , -7.347880794884119e-16 , 0.951056516295155 , -0.5877852522924724 , -0.587785252292471 , 0.9510565162951532 , 1.102182119232618e-15 , -0.951056516295155 , 0.587785252292475 , 0.5877852522924741 , -0.9510565162951521 , -1.4695761589768238e-15 , 0.951056516295153 , -0.5877852522924661 , -0.5877852522924774 , 0.951056516295153 , 0.9048270524660195 , -0.7705132427757894 , -0.24868988716485535 , 0.9822872507286886 , -0.5877852522924728 , -0.48175367410171627 , 0.9980267284282716 , -0.3681245526846787 , -0.6845471059286887 , 0.9510565162951533 , -0.12533323356430268 , -0.8443279255020163 , 0.8443279255020153 , 0.12533323356430465 , -0.951056516295155 , 0.6845471059286898 , 0.3681245526846739 , -0.9980267284282716 , 0.4817536741017115 , 0.5877852522924744 , -0.9822872507286889 , 0.24868988716485174 , 0.770513242775794 , -0.9048270524660176 , 0.844327925502015 , -0.9048270524660198 , 0.12533323356430418 , 0.7705132427757886 , -0.9510565162951534 , 0.24868988716485463 , 0.68454710592869 , -0.982287250728689 , 0.3681245526846786 , 0.5877852522924738 , -0.9980267284282714 , 0.481753674101715 , 0.48175367410171377 , -0.9980267284282718 , 0.5877852522924693 , 0.3681245526846739 , -0.9822872507286888 , 0.6845471059286896 , 0.2486898871648584 , -0.9510565162951541 , 0.770513242775794 , 0.12533323356430887 , -0.9048270524660207 , 0.8443279255020147 , 0.7705132427757893 , -0.9822872507286887 , 0.48175367410171493 , 0.368124552684678 , -0.9510565162951538 , 0.8443279255020146 , -0.1253332335643047 , -0.6845471059286887 , 0.9980267284282714 , -0.5877852522924724 , -0.24868988716485782 , 0.9048270524660202 , -0.9048270524660194 , 0.24868988716485566 , 0.5877852522924741 , -0.9980267284282716 , 0.6845471059286896 , 0.12533323356430864 , -0.8443279255020167 , 0.951056516295153 , -0.3681245526846776 , -0.48175367410172076 , 0.9822872507286896 , -0.770513242775787 , 0.6845471059286888 , -0.9980267284282716 , 0.7705132427757887 , -0.12533323356430318 , -0.5877852522924735 , 0.9822872507286889 , -0.8443279255020155 , 0.2486898871648527 , 0.48175367410171666 , -0.9510565162951539 , 0.9048270524660179 , -0.3681245526846749 , -0.3681245526846771 , 0.9048270524660189 , -0.9510565162951521 , 0.4817536741017115 , 0.24868988716485152 , -0.8443279255020167 , 0.9822872507286882 , -0.5877852522924716 , -0.12533323356430548 , 0.7705132427757941 , -0.9980267284282716 , 0.684547105928684 , 0.5877852522924732 , -0.9510565162951536 , 0.9510565162951532 , -0.5877852522924728 , -4.898587196589413e-16 , 0.5877852522924751 , -0.9510565162951538 , 0.9510565162951533 , -0.5877852522924724 , -9.797174393178826e-16 , 0.5877852522924769 , -0.951056516295155 , 0.9510565162951532 , -0.587785252292472 , -1.4695761589768238e-15 , 0.5877852522924744 , -0.9510565162951541 , 0.951056516295153 , -0.5877852522924716 , -1.959434878635765e-15 , 0.5877852522924748 , -0.9510565162951564 , 0.9510565162951506 , -0.5877852522924655 , 0.4817536741017152 , -0.8443279255020151 , 0.9980267284282716 , -0.9048270524660197 , 0.5877852522924728 , -0.1253332335643047 , -0.36812455268467664 , 0.7705132427757888 , -0.9822872507286887 , 0.9510565162951532 , -0.6845471059286847 , 0.24868988716485566 , 0.24868988716485124 , -0.6845471059286866 , 0.9510565162951552 , -0.9822872507286889 , 0.770513242775794 , -0.3681245526846776 , -0.12533323356430548 , 0.5877852522924748 , -0.9048270524660177 , 0.9980267284282709 , -0.8443279255020126 , 0.4817536741017169 , 0.36812455268467814 , -0.684547105928689 , 0.9048270524660192 , -0.9980267284282716 , 0.9510565162951534 , -0.7705132427757901 , 0.4817536741017152 , -0.12533323356430268 , -0.24868988716485438 , 0.587785252292474 , -0.8443279255020164 , 0.9822872507286882 , -0.9822872507286889 , 0.8443279255020151 , -0.5877852522924661 , 0.24868988716485174 , 0.12533323356430182 , -0.48175367410171455 , 0.7705132427757941 , -0.9510565162951543 , 0.9980267284282718 , -0.9048270524660174 , 0.6845471059286837 , -0.36812455268468347 , 0.24868988716485524 , -0.4817536741017161 , 0.6845471059286884 , -0.8443279255020161 , 0.9510565162951532 , -0.9980267284282716 , 0.9822872507286884 , -0.9048270524660179 , 0.7705132427757877 , -0.5877852522924751 , 0.3681245526846748 , -0.1253332335643057 , -0.12533323356429796 , 0.36812455268468075 , -0.5877852522924774 , 0.770513242775794 , -0.9048270524660207 , 0.9822872507286896 , -0.9980267284282716 , 0.9510565162951551 , -0.8443279255020126 , 0.6845471059286837 , -0.48175367410171366 , 0.24868988716485765 , 0.12533323356430454 , -0.24868988716485535 , 0.368124552684678 , -0.48175367410171627 , 0.5877852522924751 , -0.6845471059286887 , 0.7705132427757888 , -0.8443279255020163 , 0.9048270524660202 , -0.951056516295155 , 0.9822872507286895 , -0.9980267284282716 , 0.9980267284282718 , -0.9822872507286889 , 0.9510565162951509 , -0.9048270524660176 , 0.8443279255020186 , -0.770513242775787 , 0.684547105928684 , -0.5877852522924655 , 0.4817536741017169 , -0.36812455268467026 , 0.24868988716485765 , -0.12533323356430429};

#elif (RHO_NXT==16)
  __device__ real M[16*16]= {0.18374951781657034 , 0.3612416661871529 , 0.5264321628773557 , 0.6736956436465572 , 0.7980172272802395 , 0.8951632913550623 , 0.961825643172819 , 0.9957341762950345 , 0.9957341762950346 , 0.961825643172819 , 0.8951632913550626 ,0.7980172272802396 , 0.6736956436465571 , 0.5264321628773561 , 0.3612416661871533 , 0.18374951781657037 , 0.3612416661871529 , 0.6736956436465572 , 0.8951632913550623 , 0.9957341762950345 , 0.961825643172819 , 0.7980172272802396 , 0.5264321628773561 , 0.18374951781657037 , -0.18374951781657015 , -0.5264321628773558 , -0.7980172272802388 , -0.961825643172819 , -0.9957341762950345 , -0.8951632913550626 , -0.6736956436465578 , -0.361241666187153 , 0.5264321628773557 , 0.8951632913550623 , 0.9957341762950346 , 0.7980172272802396 , 0.3612416661871533 , -0.18374951781657015 , -0.6736956436465572 , -0.961825643172819 , -0.961825643172819 , -0.6736956436465578 , -0.18374951781657092 , 0.3612416661871526 , 0.7980172272802399 , 0.9957341762950345, 0.8951632913550635 , 0.5264321628773563 , 0.6736956436465572 , 0.9957341762950345 , 0.7980172272802396 , 0.18374951781657037 , -0.5264321628773558 , -0.961825643172819 , -0.8951632913550626 , -0.361241666187153 , 0.3612416661871526 , 0.8951632913550623 , 0.9618256431728196 , 0.5264321628773563 , -0.1837495178165712 , -0.7980172272802387 , -0.9957341762950347 , -0.6736956436465573 , 0.7980172272802395 , 0.961825643172819 , 0.3612416661871533 , -0.5264321628773558 , -0.9957341762950346 , -0.6736956436465578 , 0.18374951781656956 , 0.8951632913550623 , 0.8951632913550627 , 0.18374951781657017 , -0.6736956436465568 , -0.9957341762950347 , -0.5264321628773548 , 0.3612416661871515 , 0.9618256431728189 , 0.7980172272802394 , 0.8951632913550623 , 0.7980172272802396 , -0.18374951781657015 , -0.961825643172819 , -0.6736956436465578 , 0.3612416661871526 , 0.9957341762950345 , 0.5264321628773563 , -0.5264321628773557 , -0.9957341762950347 , -0.3612416661871541 , 0.6736956436465567 , 0.9618256431728187 , 0.18374951781657042 , -0.7980172272802365 , -0.8951632913550628 , 0.961825643172819 , 0.5264321628773561 , -0.6736956436465572 , -0.8951632913550626 , 0.18374951781656956 , 0.9957341762950345 , 0.361241666187154 , -0.7980172272802387 ,-0.7980172272802393 , 0.3612416661871515 , 0.9957341762950347 , 0.18374951781657042 , -0.8951632913550638 , -0.6736956436465589 , 0.5264321628773523 , 0.9618256431728197 , 0.9957341762950345 , 0.18374951781657037 , -0.961825643172819 , -0.361241666187153 , 0.8951632913550623 , 0.5264321628773563 , -0.7980172272802387 , -0.6736956436465573 , 0.6736956436465567 , 0.7980172272802394 , -0.5264321628773524 , -0.8951632913550628 , 0.3612416661871546 , 0.9618256431728197 , -0.18374951781656723 , -0.9957341762950346 , 0.9957341762950346 , -0.18374951781657015 , -0.961825643172819 , 0.3612416661871526 , 0.8951632913550627 , -0.5264321628773557 , -0.7980172272802393 , 0.6736956436465567 , 0.6736956436465588 , -0.7980172272802386 , -0.5264321628773566 , 0.8951632913550622 , 0.36124166618715275 , -0.9618256431728193 , -0.1837495178165725 , 0.9957341762950344 , 0.961825643172819 , -0.5264321628773558 , -0.6736956436465578 , 0.8951632913550623 , 0.18374951781657017 , -0.9957341762950347 , 0.3612416661871515 , 0.7980172272802394 , -0.7980172272802386 , -0.36124166618715264 , 0.9957341762950344 , -0.18374951781656723 , -0.8951632913550613 , 0.6736956436465549 , 0.526432162877357 , -0.9618256431728192 , 0.8951632913550626 , -0.7980172272802388 , -0.18374951781657006 , 0.9618256431728196 , -0.673695643646558 , -0.3612416661871524 , 0.9957341762950345 , -0.5264321628773524 , -0.5264321628773566 , 0.9957341762950347 , -0.3612416661871512 , -0.6736956436465564 , 0.9618256431728203 , -0.18374951781657048 , -0.7980172272802418 , 0.8951632913550588 , 0.7980172272802396 , -0.961825643172819 , 0.3612416661871526 , 0.5264321628773563 , -0.9957341762950347 , 0.6736956436465567 , 0.18374951781657042 , -0.8951632913550628 , 0.8951632913550622, -0.18374951781656723 , -0.673695643646559 , 0.9957341762950344 , -0.5264321628773581 , -0.3612416661871531 , 0.9618256431728218 , -0.7980172272802382 , 0.6736956436465571 , -0.9957341762950345 , 0.7980172272802394 , -0.1837495178165712 , -0.5264321628773564 , 0.9618256431728192 , -0.8951632913550622 , 0.3612416661871546 , 0.36124166618715275 , -0.8951632913550629 , 0.9618256431728183 , -0.5264321628773551 , -0.18374951781656926 , 0.7980172272802398 , -0.9957341762950344 , 0.6736956436465599, 0.5264321628773561 , -0.8951632913550626 , 0.9957341762950345 , -0.7980172272802387 , 0.3612416661871515 , 0.18374951781657042 , -0.6736956436465589 , 0.9618256431728197 , -0.9618256431728193 , 0.6736956436465549 , -0.18374951781656698 , -0.3612416661871531 , 0.7980172272802355 , -0.9957341762950349 , 0.8951632913550587 , -0.5264321628773516 , 0.3612416661871533 , -0.6736956436465578 , 0.8951632913550627 , -0.9957341762950347 , 0.9618256431728189 , -0.7980172272802386 , 0.5264321628773553 ,-0.18374951781656723 , -0.1837495178165725 , 0.526432162877357 , -0.7980172272802418 , 0.9618256431728199 , -0.995734176295035 , 0.8951632913550618 , -0.6736956436465571 , 0.36124166618714704 , 0.18374951781657037 , -0.361241666187153 , 0.5264321628773563 , -0.6736956436465573 , 0.7980172272802394 , -0.8951632913550628 , 0.9618256431728197 , -0.9957341762950346 , 0.9957341762950344 , -0.9618256431728192 , 0.8951632913550588 , -0.7980172272802382 , 0.6736956436465599 , -0.5264321628773516 , 0.36124166618714704 , -0.18374951781656976};

#endif


/************************************************************
  Many optimization options for SOR are defined in CUPOT.h
************************************************************/


// variables reside in constant memory
#include "CUDA_ConstMemory.h"

extern __shared__  unsigned char shared_mem[];

__device__ uint Rhoid_3Dto1D(uint x, uint y, uint z, uint N, uint XYZ){

  if (XYZ==0) return __umul24(z , N*N) + __umul24(y , N) + x;
  if (XYZ==1) return __umul24(z , N*N) + __umul24(x , N) + y;
  if (XYZ==2) return __umul24(x , N*N) + __umul24(z , N) + y;

  return 0;
}

__device__ void DST_Scheme(const uint ID,
			   uint Nslab,
			   real *Rho_Array,
			   typename FFT_DST::workspace_type workspace,
			   uint XYZ){

  uint t,Rhoid_x,Rhoid_y,Rhoid_z,Rhoid,Rhoid_r;
  uint N2 = cufftdx::size_of<FFT_DST>::value;
  uint N = (cufftdx::size_of<FFT_DST>::value / 2) - 1 ;
  uint NC = (cufftdx::size_of<FFT_DST>::value / 2) + 1 ;
  uint Nstride = N/Nslab;
  uint stride = blockDim.x * blockDim.y;
  real c;


  for (int step=0; step<RHO_NXT/2; step++){

    t = ID ;
    stride = blockDim.x * blockDim.y;
    do 
    {

     Rhoid_z = t / (N*N);
     Rhoid_y = t / N % N;
     Rhoid_x = t % N ;

     //     if (Rhoid_z < 1){
     Rhoid =  Rhoid_3Dto1D( Rhoid_x, 
			    Rhoid_y, 
			    Rhoid_z + step *2,
			    N,XYZ);
       
     reinterpret_cast<real_type*>(shared_mem)[t ]  =   Rho_Array[Rhoid];
     t += stride;
     
    } while (t < N*N*N/RHO_NXT*2);
    __syncthreads();
     
    t=ID;
    do 
    {

     Rhoid_z = t / (N*N);
     Rhoid_y = t / N % N;
     Rhoid_x = t % N ;
     Rhoid =  Rhoid_3Dto1D( Rhoid_x, 
			    Rhoid_y, 
			    Rhoid_z + step *2,
			    N,XYZ);

     c = 0;
     for (int e=0; e<N; e++)
       c += reinterpret_cast<real_type*>(shared_mem)[(Rhoid_z * N + Rhoid_y)*N + e] * M[Rhoid_x * N + e];
     
     Rho_Array[Rhoid] = c * 2.0;
     t += stride;

    } while (t < N*N*N/RHO_NXT*2);
    __syncthreads();

  } //   for (int step=0; step<RHO_NXT; step++){

    __syncthreads();
}

__device__ void Assign_sFPot(   real TempFPot,int FID,
				int bid,
				int FIDxx, int FIDyy,int FIDzz)

{
  
    int  FIDz = FID /(POT_NXT_F*POT_NXT_F);
    int  FIDy = (FID /POT_NXT_F) % POT_NXT_F ;
    int  FIDx = FID % POT_NXT_F;
    
    if (FIDx==0         and FIDy >= 1 and FIDy <= RHO_NXT and FIDz >= 1 and FIDz <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 0 + (FIDz-1)*RHO_NXT + FIDy-1 ] = TempFPot; 
    if (FIDy==0         and FIDx >= 1 and FIDx <= RHO_NXT and FIDz >= 1 and FIDz <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 1 + (FIDz-1)*RHO_NXT + FIDx-1] = TempFPot;
    if (FIDz==0         and FIDx >= 1 and FIDx <= RHO_NXT and FIDy >= 1 and FIDy <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 2 + (FIDy-1)*RHO_NXT + FIDx-1] = TempFPot;
      
    if (FIDx==RHO_NXT+1 and FIDy >= 1 and FIDy <= RHO_NXT and FIDz >= 1 and FIDz <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 3 + (FIDz-1)*RHO_NXT + FIDy-1] = TempFPot;
    if (FIDy==RHO_NXT+1 and FIDx >= 1 and FIDx <= RHO_NXT and FIDz >= 1 and FIDz <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 4 + (FIDz-1)*RHO_NXT + FIDx-1] = TempFPot;
    if (FIDz==RHO_NXT+1 and FIDx >= 1 and FIDx <= RHO_NXT and FIDy >= 1 and FIDy <= RHO_NXT) reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 5 + (FIDy-1)*RHO_NXT + FIDx-1] = TempFPot;

    //    if (FIDx==0 and FIDy-1==0 and FIDz-1==0 and bid==0) printf("Pot=%f %d %d %d %d\n",TempFPot,FID,FIDxx,FIDyy,FIDzz);

}
//-------------------------------------------------------------------------------------------------------
// Function    :  CUPOT_PoissonSolver_SOR
// Description :  GPU Poisson solver using the SOR scheme
//
// Note        :  1. Take advantage of shared memory
//                2. Prefix "g" for pointers pointing to the "Global" memory space
//                   Prefix "s" for pointers pointing to the "Shared" memory space
//                3. Each patch requires about 3.1*10^6 FLOPS (including the gravity solver)
//                   --> 133 GFLOPS is achieved in one C2050 GPU
//                4. Reference: Numerical Recipes, Chapter 20.5
//                5. Chester Cheng has implemented the SOR_USE_SHUFFLE and SOR_USE_PADDING optimizations, which
//                   greatly improve performance for PATCH_SIZE=8 && POT_GHOST_SIZE=5
//                6. Typically, the number of iterations required to reach round-off errors is 20 ~ 25 (single precision)
//                   for PATCH_SIZE=8 && POT_GHOST_SIZE=5
//
// Padding     :  Below shows how bank conflict is eliminated by padding.
//
//                Example constants :
//                      POT_NXT_F = 18                       // The number of floating point elements per row
//                      POT_PAD   = 16 - (18 * 2 % 32) = 12  // number of floating point elements that needs to be added
//                                                           // within thread groups
//
//                We now show how shared memory (s_FPot array) is accessed by a warp in residual evaluation.
//
//                Before Padding:
//                Thread number   |  Accessed shared memory bank
//                      00 ~ 07   |    | 01 |    | 03 |    | 05 |    | 07 |    | 09 |    | 11 |    | 13 |    | 15 |    |    |
//                      08 ~ 15   |    |    | 02 |    | 04 |    | 06 |    | 08 |    | 10 |    | 12 |    | 14 |    | 16 |    |
//                      16 ~ 23   |    | 05 |    | 07 |    | 09 |    | 11 |    | 13 |    | 15 |    | 17 |    | 19 |    |    |
//                      24 ~ 31   |    |    | 06 |    | 08 |    | 10 |    | 12 |    | 14 |    | 16 |    | 18 |    | 20 |    |
//
//                After Padding:
//                Thread number   |  Accessed shared memory bank
//                      00 ~ 07   |    | 01 |    | 03 |    | 05 |    | 07 |    | 09 |    | 11 |    | 13 |    | 15 |    |    |
//                      08 ~ 15   |    |    | 02 |    | 04 |    | 06 |    | 08 |    | 10 |    | 12 |    | 14 |    | 16 |    |
//                                ----------------- PAD 12 FLOATING POINTS HERE !!!!! ---------------------------------------
//                      16 ~ 23   |    | 17 |    | 19 |    | 21 |    | 23 |    | 25 |    | 27 |    | 29 |    | 31 |    |    |
//                      24 ~ 31   |    |    | 18 |    | 20 |    | 22 |    | 24 |    | 26 |    | 28 |    | 30 |    | 00 |    |
//
//
//                Additional Notes for Padding:
//                      1. When threads 08 ~ 15 access the elements below them (+y direction), we have to skip the padded
//                         elements. Same for when threads 16~23 access the elements above them (-y direction).
//                      2. For every warp we need to pad #PAD_POT floating point elements. Each xy plane has 4 warps working
//                         on it, so for each xy plane we need to pad #4*PAD_POT floating point elements.
//
//                 
// Parameter   :  g_Rho_Array     : Global memory array to store the input density
//                g_Pot_Array_In  : Global memory array storing the input "coarse-grid" potential for
//                                  interpolation
//                g_Pot_Array_Out : Global memory array to store the output potential
//                Min_Iter        : Minimum # of iterations for SOR
//                Max_Iter        : Maximum # of iterations for SOR
//                Omega_6         : Omega / 6
//                Const           : (Coefficient in front of the RHS in the Poisson eq.) / dh^2
//                IntScheme       : Interpolation scheme for potential
//                                  --> currently supported schemes include
//                                      INT_CQUAD : conservative quadratic interpolation
//                                      INT_QUAD  : quadratic interpolation
//---------------------------------------------------------------------------------------------------
//                           
__global__ void CUPOT_PoissonSolver_DST(       real g_Rho_Array    [][ RHO_NXT*RHO_NXT*RHO_NXT ], // RHO_NXT = 16 24 
                                               real g_Pot_Array_In [][ POT_NXT*POT_NXT*POT_NXT ], // POT_NXT = 12 16
                                               real g_Pot_Array_Out[][ GRA_NXT*GRA_NXT*GRA_NXT ], // GRA_NXT = 12 20
                                         const real Const, 
					 const IntScheme_t IntScheme,
					 typename FFT_DST::workspace_type workspace)
{
  //    printf("gridDim.x=%d gridDim.y=%d\n",gridDim.x ,gridDim.y);
  //  printf("%d %d %d %d\n",RHO_NXT, POT_NXT, GRA_NXT, POT_NXT_F);

  // const uint       input_size       = FFT::ffts_per_block * Nslab * cufftdx::size_of<FFT>::value;
  // real_type input_data[ Maxbid ][ input_size ];
  // __syncthreads();

  // const uint          output_size       = FFT::ffts_per_block * Nslab * (cufftdx::size_of<FFT>::value / 2 + 1);
  // complex_type output_data[ Maxbid ][ output_size ];
  // //  printf("output_size=%d\n",output_size);

  //   extern __shared__  unsigned char shared_mem[];


   const uint bid       = blockIdx.x;
   const uint tid_x     = threadIdx.x;
   const uint tid_y     = threadIdx.y;
   const uint tid_z     = threadIdx.z;
   const uint bdim_x    = blockDim.x;
   const uint bdim_y    = blockDim.y;
   const uint bdim_z    = blockDim.z;
   const uint ID        = __umul24( tid_z, __umul24(bdim_x,bdim_y) ) + __umul24( tid_y, bdim_x ) + tid_x;
   const uint dx        = 1;
   const uint dy        = POT_NXT_F;
   const uint dz        = POT_NXT_F*POT_NXT_F;
   const uint DispEven  = ( tid_y + tid_z ) & 1;
   const uint DispOdd   = DispEven^1;
   const uint DispFlip  = bdim_z & 1;
   const uint RhoID0    = __umul24( tid_z, RHO_NXT*RHO_NXT ) + __umul24( tid_y, RHO_NXT )+ ( tid_x << 1 );
   const uint dRhoID    = __umul24( bdim_z, RHO_NXT*RHO_NXT );
#  ifdef SOR_USE_PADDING
   const uint dPotID    = __umul24( bdim_z, POT_NXT_F*POT_NXT_F + POT_PAD*4 );
   const uint warpID    = ID % WARP_SIZE;
   const uint pad_dy_0  = ( warpID >=  8 && warpID <= 15 ) ? dy + POT_PAD : dy;    //
   const uint pad_dy_1  = ( warpID >= 16 && warpID <= 23 ) ? dy + POT_PAD : dy;    // please refer to the Padding notes above!
   const uint pad_dz    = dz + POT_PAD*4;                                          //
   const uint pad_pot   = ( tid_y < 2 ) ? 0 : POT_PAD*((tid_y-2)/4 + 1);
#  else
   const uint dPotID    = __umul24( bdim_z, POT_NXT_F*POT_NXT_F );
   const uint pad_dy_0  = dy;
   const uint pad_dy_1  = dy;
   const uint pad_dz    = dz;
   const uint pad_pot   = 0;
#  endif
   const uint PotID0    = pad_pot + __umul24( 1+tid_z, pad_dz ) + __umul24( 1+tid_y, dy ) + ( tid_x << 1 ) + 1;

   uint t, s_index;
   uint s_id,s_idr, stride;
   uint Rhoid, Rhoid_x,Rhoid_y,Rhoid_z;

// #  ifdef SOR_CPOT_SHARED
//    __shared__ real s_CPot[ POT_NXT  *POT_NXT  *POT_NXT   ];
//    printf("have s_CPot ??\n");
//    printf("POT_NXT=%d\n"POT_NXT);
// #  endif


// a1. load the fine-grid density into the shared memory
// -----------------------------------------------------------------------------------------------------------


// a2. load the coarse-grid potential into the shared memory
// -----------------------------------------------------------------------------------------------------------
#  ifdef SOR_CPOT_SHARED
   t = ID;
   do {  s_CPot[t] = g_Pot_Array_In[bid][t];    t += POT_NTHREAD; }     while ( t < POT_NXT*POT_NXT*POT_NXT );
   __syncthreads();
#  else
   const real *s_CPot = g_Pot_Array_In[bid];
#  endif

// b. evaluate the "fine-grid" potential by interpolation (as the initial guess and the B.C.)
// -----------------------------------------------------------------------------------------------------------
   const int N_CSlice = POT_NTHREAD / ( (POT_NXT-2)*(POT_NXT-2) );

   if ( ID < N_CSlice*(POT_NXT-2)*(POT_NXT-2) )
   {
      const real Const_8   = 1.0/8.0;
      const real Const_64  = 1.0/64.0;
      const real Const_512 = 1.0/512.0;

      const int Cdx  = 1;
      const int Cdy  = POT_NXT;
      const int Cdz  = POT_NXT*POT_NXT;
      const int CIDx = 1 + ID % ( POT_NXT-2 );
      const int CIDy = 1 + (  ID % ( (POT_NXT-2)*(POT_NXT-2) )  ) / ( POT_NXT-2 );
      const int CIDz = 1 + ID / ( (POT_NXT-2)*(POT_NXT-2) );
      int       CID  = __mul24( CIDz, Cdz ) + __mul24( CIDy, Cdy ) + __mul24( CIDx, Cdx );
      const int Fdx  = 1;
      const int Fdy  = POT_NXT_F;
      const int FIDx = ( (CIDx-1)<<1 ) - POT_USELESS;
      const int FIDy = ( (CIDy-1)<<1 ) - POT_USELESS;
      int       FIDz = ( (CIDz-1)<<1 ) - POT_USELESS;
#     ifdef SOR_USE_PADDING
      const int Fpad = ( FIDy < 3 ) ? 0 : POT_PAD*((FIDy-3)/4 + 1);    // padding logic
      const int Fdz  = POT_NXT_F*POT_NXT_F + POT_PAD*4;                // added padding
#     else
      const int Fpad = 0;
      const int Fdz  = POT_NXT_F*POT_NXT_F;
#     endif
      int       FID  = Fpad + __mul24( FIDz, Fdz ) + __mul24( FIDy, Fdy ) + __mul24( FIDx, Fdx );

      real TempFPot1, TempFPot2, TempFPot3, TempFPot4, TempFPot5, TempFPot6, TempFPot7, TempFPot8;
      real Slope_00, Slope_01, Slope_02, Slope_03, Slope_04, Slope_05, Slope_06, Slope_07;
      real Slope_08, Slope_09, Slope_10, Slope_11, Slope_12;
      int  Idx, Idy, Idz, ii, jj, kk;


      for (int z=CIDz; z<POT_NXT-1; z+=N_CSlice)
      {
         switch ( IntScheme )
         {
            /*
            case INT_CENTRAL :
            {
               Slope_00 = (real)0.125 * ( s_CPot[CID+Cdx] - s_CPot[CID-Cdx] );
               Slope_01 = (real)0.125 * ( s_CPot[CID+Cdy] - s_CPot[CID-Cdy] );
               Slope_02 = (real)0.125 * ( s_CPot[CID+Cdz] - s_CPot[CID-Cdz] );

               TempFPot1 = s_CPot[CID] - Slope_00 - Slope_01 - Slope_02;
               TempFPot2 = s_CPot[CID] + Slope_00 - Slope_01 - Slope_02;
               TempFPot3 = s_CPot[CID] - Slope_00 + Slope_01 - Slope_02;
               TempFPot4 = s_CPot[CID] + Slope_00 + Slope_01 - Slope_02;
               TempFPot5 = s_CPot[CID] - Slope_00 - Slope_01 + Slope_02;
               TempFPot6 = s_CPot[CID] + Slope_00 - Slope_01 + Slope_02;
               TempFPot7 = s_CPot[CID] - Slope_00 + Slope_01 + Slope_02;
               TempFPot8 = s_CPot[CID] + Slope_00 + Slope_01 + Slope_02;
            }
            break; // INT_CENTRAL
            */


            case INT_CQUAD :
            {
               Slope_00 = Const_8   * ( s_CPot[CID+Cdx        ] - s_CPot[CID-Cdx        ] );
               Slope_01 = Const_8   * ( s_CPot[CID    +Cdy    ] - s_CPot[CID    -Cdy    ] );
               Slope_02 = Const_8   * ( s_CPot[CID        +Cdz] - s_CPot[CID        -Cdz] );

               Slope_03 = Const_64  * ( s_CPot[CID+Cdx    -Cdz] - s_CPot[CID-Cdx    -Cdz] );
               Slope_04 = Const_64  * ( s_CPot[CID    +Cdy-Cdz] - s_CPot[CID    -Cdy-Cdz] );
               Slope_05 = Const_64  * ( s_CPot[CID+Cdx-Cdy    ] - s_CPot[CID-Cdx-Cdy    ] );
               Slope_06 = Const_64  * ( s_CPot[CID+Cdx+Cdy    ] - s_CPot[CID-Cdx+Cdy    ] );
               Slope_07 = Const_64  * ( s_CPot[CID+Cdx    +Cdz] - s_CPot[CID-Cdx    +Cdz] );
               Slope_08 = Const_64  * ( s_CPot[CID    +Cdy+Cdz] - s_CPot[CID    -Cdy+Cdz] );

               Slope_09 = Const_512 * ( s_CPot[CID+Cdx-Cdy-Cdz] - s_CPot[CID-Cdx-Cdy-Cdz] );
               Slope_10 = Const_512 * ( s_CPot[CID+Cdx+Cdy-Cdz] - s_CPot[CID-Cdx+Cdy-Cdz] );
               Slope_11 = Const_512 * ( s_CPot[CID+Cdx-Cdy+Cdz] - s_CPot[CID-Cdx-Cdy+Cdz] );
               Slope_12 = Const_512 * ( s_CPot[CID+Cdx+Cdy+Cdz] - s_CPot[CID-Cdx+Cdy+Cdz] );

               TempFPot1 = - Slope_00 - Slope_01 - Slope_02 - Slope_03 - Slope_04 - Slope_05 + Slope_06
                           + Slope_07 + Slope_08 - Slope_09 + Slope_10 + Slope_11 - Slope_12 + s_CPot[CID];

               TempFPot2 = + Slope_00 - Slope_01 - Slope_02 + Slope_03 - Slope_04 + Slope_05 - Slope_06
                           - Slope_07 + Slope_08 + Slope_09 - Slope_10 - Slope_11 + Slope_12 + s_CPot[CID];

               TempFPot3 = - Slope_00 + Slope_01 - Slope_02 - Slope_03 + Slope_04 + Slope_05 - Slope_06
                           + Slope_07 - Slope_08 + Slope_09 - Slope_10 - Slope_11 + Slope_12 + s_CPot[CID];

               TempFPot4 = + Slope_00 + Slope_01 - Slope_02 + Slope_03 + Slope_04 - Slope_05 + Slope_06
                           - Slope_07 - Slope_08 - Slope_09 + Slope_10 + Slope_11 - Slope_12 + s_CPot[CID];

               TempFPot5 = - Slope_00 - Slope_01 + Slope_02 + Slope_03 + Slope_04 - Slope_05 + Slope_06
                           - Slope_07 - Slope_08 + Slope_09 - Slope_10 - Slope_11 + Slope_12 + s_CPot[CID];

               TempFPot6 = + Slope_00 - Slope_01 + Slope_02 - Slope_03 + Slope_04 + Slope_05 - Slope_06
                           + Slope_07 - Slope_08 - Slope_09 + Slope_10 + Slope_11 - Slope_12 + s_CPot[CID];

               TempFPot7 = - Slope_00 + Slope_01 + Slope_02 + Slope_03 - Slope_04 + Slope_05 - Slope_06
                           - Slope_07 + Slope_08 - Slope_09 + Slope_10 + Slope_11 - Slope_12 + s_CPot[CID];

               TempFPot8 = + Slope_00 + Slope_01 + Slope_02 - Slope_03 - Slope_04 - Slope_05 + Slope_06
                           + Slope_07 + Slope_08 + Slope_09 - Slope_10 - Slope_11 + Slope_12 + s_CPot[CID];
            }
            break; // INT_CQUAD

            case INT_QUAD :
            {
               TempFPot1 = TempFPot2 = TempFPot3 = TempFPot4 = (real)0.0;
               TempFPot5 = TempFPot6 = TempFPot7 = TempFPot8 = (real)0.0;

               for (int dk=-1; dk<=1; dk++)  {  Idz = dk+1;    kk = __mul24( dk, Cdz );
               for (int dj=-1; dj<=1; dj++)  {  Idy = dj+1;    jj = __mul24( dj, Cdy );
               for (int di=-1; di<=1; di++)  {  Idx = di+1;    ii = __mul24( di, Cdx );

                  TempFPot1 += s_CPot[CID+kk+jj+ii] * c_Mm[Idz] * c_Mm[Idy] * c_Mm[Idx];
                  TempFPot2 += s_CPot[CID+kk+jj+ii] * c_Mm[Idz] * c_Mm[Idy] * c_Mp[Idx];
                  TempFPot3 += s_CPot[CID+kk+jj+ii] * c_Mm[Idz] * c_Mp[Idy] * c_Mm[Idx];
                  TempFPot4 += s_CPot[CID+kk+jj+ii] * c_Mm[Idz] * c_Mp[Idy] * c_Mp[Idx];
                  TempFPot5 += s_CPot[CID+kk+jj+ii] * c_Mp[Idz] * c_Mm[Idy] * c_Mm[Idx];
                  TempFPot6 += s_CPot[CID+kk+jj+ii] * c_Mp[Idz] * c_Mm[Idy] * c_Mp[Idx];
                  TempFPot7 += s_CPot[CID+kk+jj+ii] * c_Mp[Idz] * c_Mp[Idy] * c_Mm[Idx];
                  TempFPot8 += s_CPot[CID+kk+jj+ii] * c_Mp[Idz] * c_Mp[Idy] * c_Mp[Idx];

               }}}
            }
            break; // INT_QUAD

         } // switch ( IntScheme )

//       save data to the shared-memory array.
//       Currently this part is highly diverged. However, since the interpolation takes much less time than the
//       SOR iteration does, we have not yet tried to optimize this part

         if ( FIDz >= 0 )
         {
            if ( FIDx >= 0            &&  FIDy >= 0           )   Assign_sFPot(TempFPot1, FID,bid,FIDx,FIDy,FIDz);;
            if ( FIDx <= POT_NXT_F-2  &&  FIDy >= 0           )   Assign_sFPot(TempFPot2, FID+Fdx,bid,FIDx,FIDy,FIDz);
            if ( FIDx >= 0            &&  FIDy <= POT_NXT_F-2 )   Assign_sFPot(TempFPot3, FID    +Fdy,bid,FIDx,FIDy,FIDz);
            if ( FIDx <= POT_NXT_F-2  &&  FIDy <= POT_NXT_F-2 )   Assign_sFPot(TempFPot4, FID+Fdx+Fdy  ,bid,FIDx,FIDy,FIDz);
         }
	 
         if ( FIDz <= POT_NXT_F-2 )
         {
            if ( FIDx >= 0            &&  FIDy >= 0           )   Assign_sFPot(TempFPot5, FID        +Fdz,bid,FIDx,FIDy,FIDz);
            if ( FIDx <= POT_NXT_F-2  &&  FIDy >= 0           )   Assign_sFPot(TempFPot6, FID+Fdx    +Fdz,bid,FIDx,FIDy,FIDz);
            if ( FIDx >= 0            &&  FIDy <= POT_NXT_F-2 )   Assign_sFPot(TempFPot7, FID    +Fdy+Fdz,bid,FIDx,FIDy,FIDz);
            if ( FIDx <= POT_NXT_F-2  &&  FIDy <= POT_NXT_F-2 )   Assign_sFPot(TempFPot8, FID+Fdx+Fdy+Fdz,bid,FIDx,FIDy,FIDz);
         }


	 
         CID  += __mul24(   N_CSlice, Cdz );
         FID  += __mul24( 2*N_CSlice, Fdz );
         FIDz += 2*N_CSlice;

      } // for (int z=CIDz; z<POT_NXT-1; z+=N_CSlice)
   } // if ( ID < N_CSlice*(POT_NXT-2)*(POT_NXT-2) )
   __syncthreads();
   
   float bc_xm,bc_xp,bc_ym,bc_yp,bc_zm,bc_zp;
   unsigned int N = (cufftdx::size_of<FFT_DST>::value / 2) - 1 ;
         
// allocation shared_mem = [0 a b c d e f g 0 -g -f -e -d -c -b -a ] for DST from rho = [a b c d e f g]
   real temp;
   uint Nstride = RHO_NXT/Nslab;

   t = ID;
   stride = blockDim.x * blockDim.y;

   do 
   {
     //     if (t < RHO_NXT*RHO_NXT*RHO_NXT/Nslab * (stepB+1)){
     if (t < RHO_NXT*RHO_NXT*RHO_NXT){
     Rhoid_x = t % RHO_NXT;
     Rhoid_y = t/RHO_NXT % RHO_NXT;
     Rhoid_z = t / (RHO_NXT*RHO_NXT);

     s_id = __umul24(Rhoid_z, N*N)
          + __umul24(Rhoid_y, N)
          + Rhoid_x;
	     
     // if boundary condition
     if (Rhoid_x==0)         {bc_xm = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 0 + Rhoid_y + RHO_NXT * Rhoid_z];} //[k+2][j+2][im+2];}
     else bc_xm = 0.0;
     if (Rhoid_y==0)         {bc_ym = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 1 + Rhoid_x + RHO_NXT * Rhoid_z];} //[k+2][j+2][im+2];}
     else bc_ym = 0.0;
     if (Rhoid_z==0)         {bc_zm = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 2 + Rhoid_x + RHO_NXT * Rhoid_y];} //[k+2][j+2][im+2];}
     else bc_zm = 0.0;
	 
     if (Rhoid_x==RHO_NXT-1) {bc_xp = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 3 + Rhoid_y + RHO_NXT * Rhoid_z];} //[k+2][j+2][ip+2];}
     else bc_xp = 0.0;
     if (Rhoid_y==RHO_NXT-1) {bc_yp = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 4 + Rhoid_x + RHO_NXT * Rhoid_z];} //[k+2][j+2][ip+2];}
     else bc_yp = 0.0;
     if (Rhoid_z==RHO_NXT-1) {bc_zp = reinterpret_cast<real_type*>(shared_mem)[RHO_NXT*RHO_NXT * 5 + Rhoid_x + RHO_NXT * Rhoid_y];} //[k+2][j+2][ip+2];}
     else bc_zp = 0.0;

     //     g_Rho_Array[bid][t] = t; 
     g_Rho_Array[bid][t] *= -Const; 
     g_Rho_Array[bid][t] += bc_xm + bc_ym + bc_zm + bc_xp + bc_yp + bc_zp;

     }
     t += stride;    

   } while (t < RHO_NXT*RHO_NXT*RHO_NXT);
   __syncthreads();

  // if (threadIdx.x==0 && threadIdx.y==0 && bid==0){
  //   for (int i = 0; i<RHO_NXT; i++){
  //    for (int j = 0; j<RHO_NXT; j++){
  //      for (int k = 0; k<RHO_NXT; k++){
  // 	 printf("After interp  ijk=%d %d %d ids=%d shared=%.9e \n", i,j,k,(i*RHO_NXT + j)*N + k, g_Rho_Array[bid][(i*RHO_NXT + j)*N + k]);	 
  //  }}}
  // }
  //    __syncthreads();
   

   //  ----------------------------------------------   Discret Sine Transform Poisson Solver -- START -----------------------------------------------//

   //  Forward FFT
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,0);
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,1);
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,2);

  // if (threadIdx.x==0 && threadIdx.y==0 && bid==0){
  //   for (int i = 0; i<RHO_NXT; i++){
  //    for (int j = 0; j<RHO_NXT; j++){
  //      for (int k = 0; k<RHO_NXT; k++){
  // 	 printf("After interp  ijk=%d %d %d ids=%d shared=%.9e \n", i,j,k,(i*RHO_NXT + j)*N + k, g_Rho_Array[bid][(i*RHO_NXT + j)*N + k]);	 
  //  }}}
  // }
  //    __syncthreads();

    // Poisson Eigen
  __shared__ real Eigen[RHO_NXT];
  for (int i=0; i<RHO_NXT; i++)
    Eigen[i] = 1.-COS(M_PI*(i+1)/(RHO_NXT+1));

   __syncthreads();
   
   t=ID;
   do
     {
     Rhoid_z = t /(RHO_NXT*RHO_NXT);
     Rhoid_y = (t /RHO_NXT) % RHO_NXT;
     Rhoid_x = t % RHO_NXT;
     g_Rho_Array[bid][t] /=  2. * (Eigen[Rhoid_x] + Eigen[Rhoid_y] + Eigen[Rhoid_z]) * (8*(N+1)*(N+1)*(N+1));

     t +=stride;

     } while (t < RHO_NXT*RHO_NXT*RHO_NXT);
     __syncthreads();
 
   // // reverse FFT  
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,0);
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,1);
   DST_Scheme(ID, Nslab, g_Rho_Array[bid],workspace,2);
    
  // if (threadIdx.x==0 && threadIdx.y==0 && bid==0){
  //   for (int i = 0; i<RHO_NXT; i++){
  //    for (int j = 0; j<RHO_NXT; j++){
  //      for (int k = 0; k<RHO_NXT; k++){
  // 	 printf("After interp  ijk=%d %d %d ids=%d shared=%.9e \n", i,j,k,(i*RHO_NXT + j)*N + k, g_Rho_Array[bid][(i*RHO_NXT + j)*N + k] / (8*(N+1)*(N+1)*(N+1)));	 
  //  }}}
  // }
  //    __syncthreads();

   //  ----------------------------------------------   Discret Sine Transform Poisson Solver -- END  -----------------------------------------------//

   t=ID;
   do
   {
     Rhoid_z = t /(GRA_NXT*GRA_NXT);
     Rhoid_y = (t /GRA_NXT) % GRA_NXT;
     Rhoid_x = t % GRA_NXT;

     Rhoid    =  (Rhoid_z + 2) * (RHO_NXT) * (RHO_NXT)
               + (Rhoid_y + 2) * (RHO_NXT)
               + (Rhoid_x + 2) ;

     if (t< GRA_NXT * GRA_NXT * GRA_NXT){
       g_Pot_Array_Out[bid][t] = g_Rho_Array[bid][Rhoid] ; /// (8*(N+1)*(N+1)*(N+1));
     }
     t += stride;

   } while (t< GRA_NXT * GRA_NXT * GRA_NXT);
    __syncthreads();
} // FUNCTION : CUPOT_PoissonSolver_SOR



#endif // #if ( defined GRAVITY  &&  defined GPU  &&  POT_SCHEME == SOR )
