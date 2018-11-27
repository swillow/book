#ifndef W_CONSTANTS_HPP
#define W_CONSTANTS_HPP

namespace willow {

// --- (2014 CODATA) ---
const double kB         = 1.38064852e-23; // J/K
const double au_length  = 0.52917721067;  // A
const double au_energy  = 4.359744650e-18; // J
const double au_time    = 2.418884326509e-17; // s
const double au_mass    = 9.10938356e-31;  // kg
const double amu_mass   = 1.660539040e-27; // kg
const double au_charge  = 1.6021766208e-19; // C
const double light_speed= 299792458; // m/s
const double planck_h   = 6.626070040e-34; // J s
const double avogadro   = 6.022140857e23; // mol-1
const double kcal       = 4184.0; // J

const double au_kcal    = au_energy/kcal*avogadro; // AU--> 627.509 kcal/mol
const double au_freq    = au_energy/(light_speed*planck_h*100); // 2.19475e5 cm-1

const double ang2bohr  = 1.0/au_length;  //   ! A    --> bohr
const double bohr2ang  = au_length;      //   ! bohr --> A

const double boltz     = kB/au_energy;   //    ! K in atomic unit
const double amu2au    = amu_mass/au_mass; // amu --> A.U.

const double hbar      = 1.0;

}



#endif
