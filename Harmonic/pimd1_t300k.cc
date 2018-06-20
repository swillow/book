#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <armadillo>

#include "constant.hpp"
#include "mt_random.hpp"

using namespace std;

namespace willow {

// global variables
const  int m_nnhc  = 4;
const  double kval = 0.4629344789;
static int m_nbead;
static int m_nstep;
static int m_irst;
static int m_nref;
static double m_dt;
static double m_dt_ref;
static double m_gfree_nm;

static double m_Dmorse; // (De) in Morse potential: De (1 - exp(-a*x))**2
static double m_ZPE;
static double m_omega;
static double m_omega_p2;
static double m_temp_nm;
static double m_beta_nm;
static double m_ekin_nm;

const  double delt_x = 4.e-3;

static arma::vec prob_dx;      // (0:399)

static double     m_mass;     

static arma::mat  m_tmat_nm;  // (m_nbead, m_nbead)
static arma::vec  m_fict_mass;// (m_nbead)

static arma::vec  m_rbath_nm; // (m_nnhc)
static arma::mat  m_vbath_nm; //
static arma::mat  m_qmass_nm; //

// one-dimensional system
// Cartesian Coordinate 
static arma::vec  m_pos_qm;   // (m_nbead)
static arma::vec  m_grd_qm;   // (m_nbead)

// Normal Mode
static arma::vec  m_pos_nm;   // (m_nbead)
static arma::vec  m_vel_nm;   // (m_nbead)
static arma::vec  m_grd_nm;   // (m_nbead)
static arma::vec  m_grd_nm_spr; //(m_nbead)
  

static void sample_rho ()
{

  // QM (beads)
  
  for (int ib = 0; ib < m_nbead; ++ib) {

    // distribution of beads around pos(0) 
    // \rho(x) = \varrho (x1 - X1)  
    double dx   = m_pos_qm(ib); 
    int    id1  = (int) round(dx/delt_x) + 200;
    if (id1 >= 0 && id1 < 400) prob_dx (id1) += 1;
    
  }
  
}


void nm_nhc_integrate ()
{
  // Nose-Hoover Chain Method
  
  const double dt_ref  = m_dt_ref;
  const double dt_ref2 = 0.5*dt_ref;
  const double dt_ref4 = 0.5*dt_ref2;
  const double dt_ref8 = 0.5*dt_ref4;

  // m_nnhc = 4
  arma::vec4 rbath;
  arma::vec4 vbath;
  arma::vec4 fbath;
  arma::vec4 qmass;
  
  
  // ---
  // 
  double ekin_nm = 0.0;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    
    double mass = m_fict_mass (ib);
    double v    = m_vel_nm(ib);
    
    ekin_nm += mass * v * v;
      
  }

  //---
    
  vbath = m_vbath_nm;
  rbath = m_rbath_nm;
  qmass = m_qmass_nm;

  // ihc = 0
  fbath(0) = (ekin_nm - m_gfree_nm*m_ekin_nm)/qmass(0);
    
  for (size_t ihc = 1; ihc < m_nnhc; ++ihc) {
    fbath(ihc) =
      (qmass(ihc-1)*vbath(ihc-1)*vbath(ihc-1)	- m_ekin_nm) / qmass(ihc);
  }
  
  // Update Thermostat Velocities
  
  vbath(m_nnhc-1) = vbath(m_nnhc-1) + fbath(m_nnhc-1)*dt_ref4;
    
  for (int ihc = 1; ihc < m_nnhc; ++ihc) {
    const int jhc     = m_nnhc - ihc;
    const double vfact = exp (-vbath(jhc)*dt_ref8);
    const double vtmp  = vbath(jhc-1);
    vbath(jhc-1) = vtmp*vfact*vfact + fbath(jhc-1)*vfact*dt_ref4;
  }
  
  // Update atomic velocities
  const double pvfact = exp(-vbath(0)*dt_ref2);
    
  ekin_nm  = ekin_nm*pvfact*pvfact;
    
  // Update thermostat forces
  
  fbath(0) = (ekin_nm - m_gfree_nm*m_ekin_nm)/qmass(0);
    
  for (int ihc = 0; ihc < m_nnhc; ++ihc) {
    rbath(ihc) += vbath(ihc)*dt_ref2;
  }
  
  // Update Thermostat velocities
  
  for (int ihc = 0; ihc < m_nnhc-1; ++ihc) {
    const double vfact = exp(-vbath(ihc+1)*dt_ref8);
    const double vtmp  = vbath(ihc);
    
    vbath(ihc) = vtmp*vfact*vfact + fbath(ihc)*vfact*dt_ref4;
    fbath(ihc+1) =
      (qmass(ihc)*vbath(ihc)*vbath(ihc) - m_ekin_nm)/qmass(ihc+1);
  }
  
  vbath(m_nnhc-1) += fbath(m_nnhc-1)*dt_ref4;
  
  
  // backup
  m_vbath_nm = vbath;
  m_rbath_nm = rbath;
  
  // update velocities of normal modes
  for (int ib = 1; ib < m_nbead; ++ib)
    m_vel_nm(ib) *= pvfact;

  
}



void nm_pos_update ()
{
  
  m_pos_nm += m_dt_ref * m_vel_nm;

}




void nm_grad_spring ()
{
  
  //
  // centroid gradient is zero
  //
  m_grd_nm_spr(0) = 0.0;
  
  //
  // Gradients from the Springs between 'neighboring' beads
  //
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    
    const double fact = m_fict_mass(ib)*m_omega_p2;

    m_grd_nm_spr(ib) = fact * m_pos_nm(ib);
  }
  
  
}





void nm_pos_trans ()
{
  //
  // normal mode (nm) ---> Cartesian (qm)
  // pos_nm --->  pos_qm
  
  for (int ib = 0; ib < m_nbead; ++ib) {
    
    double pos_x = 0.0;
    
    for (int jb = 0; jb < m_nbead; ++jb) {
      pos_x += m_tmat_nm(ib,jb)*m_pos_nm(jb);
    }
    
    m_pos_qm(ib) = pos_x;
  }

}




void nm_grad_trans ()
{

  m_grd_nm.zeros();
  
  for (size_t ib = 0; ib < m_nbead; ++ib) {
    for (size_t jb = 0; jb < m_nbead; ++jb) {
      m_grd_nm(ib) += m_tmat_nm(jb,ib)*m_grd_qm(jb);
    }
  }

}




void nm_pos_init () 
{

  {// centroid particle
    m_pos_nm(0) = 0.0;
  }

  { // beads : normal mode
    const double dbead = m_nbead;
    const double usigma = 0.02*ang2bohr; // sigma_x = 0.02 A
    
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {
      
      double mass05 = sqrt(m_fict_mass(ib));
      m_pos_nm(ib) = usigma*rnd::gaus_dev()/mass05;
      
    }
  } // beads

}


void nm_vel_init ()
{

  // ---- centroid ----
  {
    // Here, vel_nm(ib = 0) zero
    m_vel_nm(0) = 0.0;
  }
  
  { // velocities for bead particles (or nm-mode particles)

    for (size_t ib = 1; ib < m_nbead; ++ib) {
	
      double mass   = m_fict_mass(ib);
      double vsigma = sqrt (m_ekin_nm/mass);

      double v = vsigma*rnd::gaus_dev();

      m_vel_nm(ib) = v; 
	
    }
      
    // --- one particle on one-dimensional harmonic oscillator
    // no correction of translational and rotational motions 
    
    // Scale Velocity 
    double ekin_nm = 0.0;
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {
      
      double mass  = m_fict_mass(ib);
      double v     = m_vel_nm(ib);
      ekin_nm += mass * v*v;
      
    }

    double temp  = ekin_nm / (m_gfree_nm*boltz);
    double scale = sqrt(m_temp_nm / temp);
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {

      m_vel_nm(ib) *= scale;
	
    }
	
    
  } // velocities for bead particles

}



void nm_vel_update ()
{

  // (ib = 0) belongs to the centroid velocity

  
  //---
  const double dt2 = 0.5*m_dt;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    double mass   = m_fict_mass(ib); 
    double factor = dt2/mass;
    m_vel_nm(ib) -= factor*m_grd_nm(ib);
  }

}




void nm_vel_spring_update ()
{

  double dt2 = 0.5*m_dt_ref;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    
    double mass   = m_fict_mass(ib); 
    double factor = dt2/mass;
    m_vel_nm(ib) -=
      factor*m_grd_nm_spr(ib);
  }


}


double nm_pot_grad ()
{
  
  // beads: normal mode ---> cartesian
  //        pos_nm ----> pos_qm
  nm_pos_trans ();
  
  double u_vib = 0.0;
  
  m_grd_qm.zeros();

  //
  // force constant: reduced_mass*omega*omega
  //
  
  //double k_val = m_mass*m_omega*m_omega;

  //std::cout << k_val << std::endl;

  
  
  // Harmonic Oscilltor
  // U = 0.5 * k * x**2
  //
  for (size_t ib = 0; ib < m_nbead; ib++) {
    
    // call your potential.
    double dx      = m_pos_qm(ib); 
    double en_harm = 0.5*kval*dx*dx;
    m_grd_qm(ib)   =  kval*dx;
      
    u_vib += en_harm;
  } // ib
    

  // ---
  double d_nbead = m_nbead;
  u_vib /= d_nbead;
  
  m_grd_qm /= d_nbead;
  
  //
  // cartesian gradient ---> normal mode gradient
  //
  nm_grad_trans ();
  
  
  return u_vib;
  
}


void nm_report (const int& istep,
		const double& u_vib,
		double& E_vib) 
{
  // kinetic energy for beads
  double ekin_nm = 0.0;
  
  for (int ib = 1; ib < m_nbead; ++ib) {
    double mass = m_fict_mass(ib);
    double v    = m_vel_nm(ib); 
    ekin_nm += mass*v*v;
  }
  

  ekin_nm = 0.5*ekin_nm;
  double temp_nm = 2.0*ekin_nm/(m_gfree_nm*boltz);

  // Harmonic Potential of springs between neighboring beads

  double qkin_nm = 0.0;
  for (int ib = 1; ib < m_nbead; ++ib) {
    double fact  = 0.5*m_fict_mass(ib)*m_omega_p2;
    double q     = m_pos_nm(ib);
    qkin_nm += fact*q*q;
  
  }
  
  double ebath_nm = 0.0;

  arma::vec4 qmass = m_qmass_nm;
  ebath_nm += 0.5*qmass(0)*m_vbath_nm(0)*m_vbath_nm(0);
  ebath_nm += m_gfree_nm*m_ekin_nm*m_rbath_nm(0);

  for (int ihc = 1; ihc < m_nnhc; ++ihc) {
    ebath_nm += 0.5*qmass(ihc)*m_vbath_nm(ihc)*m_vbath_nm(ihc);
    ebath_nm += m_ekin_nm*m_rbath_nm(ihc);
  }

  E_vib = qkin_nm + u_vib; // <E_vib> = E_ZPE
  double H_sys = ekin_nm + E_vib; // Hamiltonian of the system
  double H_tot = H_sys + ebath_nm; // Total H.

  // unit convert
  E_vib *= au_kcal;
  H_sys *= au_kcal;
  H_tot *= au_kcal;

  printf (" %8d %14.6f %14.6f %14.6f %14.6f %10.2f \n",
	  (istep+1), H_tot, H_sys, E_vib, 
	  u_vib*au_kcal, temp_nm);
  
  fflush (stdout);
  
}



void read_restart_nm (int& istep0)
{

  // read a restart file
  std::string str_rst;

  std::ifstream ifs_rst ("pimdrr.sav");
  assert (ifs_rst.good());
  
  std::ostringstream oss;
  
  oss << ifs_rst.rdbuf();
  
  str_rst = oss.str();

  
  std::istringstream is (str_rst);

  std::string line;
  std::getline (is, line); // istep
  std::istringstream istep_ss (line);
  istep_ss >> istep0;
  
  // ib = 0 is for centroids
  for (int ib = 1; ib < m_nbead; ++ib) {
    std::getline (is, line);
    std::istringstream iss_pos (line);
    
    iss_pos >> m_pos_nm(ib); 
  
    std::getline (is, line);
    std::istringstream iss_vel (line);

    iss_vel >> m_vel_nm(ib); 
  }

  for (int ihc = 0; ihc < m_nnhc; ++ihc) {
    std::getline (is,line);
    std::istringstream iss (line);

    iss >> m_rbath_nm(ihc); 
  }

  // position and velocity of the centroid particle
  m_pos_nm(0) = 0.0; 
  m_vel_nm(0) = 0.0; 
  
}


void write_restart_nm (const int& istep)
{

  FILE *ofs_rst = fopen ("pimdrr.sav", "w");

  fprintf( ofs_rst, " %10d \n", istep+1);

  // ib = 0 is for centroids
  for (int ib = 1; ib < m_nbead; ++ib) {
    fprintf (ofs_rst, "  %E   \n", m_pos_nm(ib) );
    fprintf (ofs_rst, "  %E   \n", m_vel_nm(ib) );
  }

  for (int ihc = 0; ihc < m_nnhc; ++ihc) {
    fprintf (ofs_rst, "  %E   %E   %E \n",
	     m_rbath_nm(ihc),
	     m_vbath_nm(ihc),
	     m_qmass_nm(ihc) );
  }
  
  fclose (ofs_rst);

}


void write_prob_bin (int& nsamp)
{

  double zeta2   = m_mass*m_omega; // (unit 1/bohr**2)
  
  arma::vec p_dx = prob_dx/(m_nbead*(nsamp+1)*delt_x);
  
  std::ofstream ofs_rho ("prob_bin_rho.dat");
  // psi_0
  // zeta2 = mass*omega*(x*x)
  // debug
  const double zt    = sqrt(zeta2/M_PI); 

  for (int ib = 0; ib < 400; ++ib) {
    double  x  = (ib - 200)*delt_x;
    double  x2 =  x* x;
    double rho =  zt*exp(-zeta2*x2);
    ofs_rho << x << "   "  << p_dx(ib)
	   << "  " << rho << endl;
  }

  ofs_rho.close();

}


void wpimd_run ()
{

  //
  // This is a sample code,
  // in which WPIMD is running at T (thermal temperature) = 0 K.
  // 
  
  int istep0 = 0;

  if (m_irst == 1) {
    read_restart_nm (istep0);
  }
  
  // initial gradients and potential energy
  double u_vib = nm_pot_grad ();

  double e_vib = 0.0;
  double sum_e_vib = 0.0;
  double sum_u_vib = 0.0;
  int    ncount    = 0;
  
  nm_grad_spring ();

  if (m_irst == 0)
    nm_report (istep0-1, u_vib, e_vib);

  for (int istep = 0; istep < m_nstep; ++istep) {

    nm_vel_update ();

    for (int iref = 0; iref < m_nref; ++iref) {
      nm_nhc_integrate ();
      nm_vel_spring_update();
      nm_pos_update ();
      nm_grad_spring();
      nm_vel_spring_update();
      nm_nhc_integrate ();
    }

    u_vib   = nm_pot_grad ();
    
    sample_rho ();
    
    nm_vel_update ();
    
    if ( (istep+1)%500 == 0) {
      nm_report (istep+istep0, u_vib, e_vib);
      sum_e_vib += e_vib;
      sum_u_vib += u_vib;
      ncount++;
    }
    
    if ( (istep+1)%5000 == 0) {
      write_restart_nm   (istep+istep0);
      write_prob_bin (istep);
    }
    
  }


  cout << "AVE E_zpe " << m_ZPE*au_kcal << endl;
  cout << "AVE E_vib " << sum_e_vib / ncount << endl;
  cout << "AVE U_vib " << sum_u_vib*au_kcal / ncount << endl;
  

}




void wpimd_init ()
{

  // one-dimensional system
  m_gfree_nm = m_nbead;
  
  // ZPE = 0.5 * hbar * omega
  // ZPE(au) = 0.5 * omega
  //m_omega    = 2.0*m_ZPE;
  
  m_omega    = std::sqrt(kval/m_mass);
  m_ZPE      = 0.5 * m_omega;

  // Eq. (14)
  // temperature for the bead motions
  m_temp_nm = 300.0;

  double dbead = m_nbead;
  double omega_p = sqrt(dbead)*boltz*m_temp_nm;
  m_omega_p2 = omega_p * omega_p;
  m_beta_nm  = 1.0 / (boltz*m_temp_nm);
  m_ekin_nm  = boltz*m_temp_nm;
  
  // mem alloc

  m_tmat_nm   = arma::mat (m_nbead, m_nbead, arma::fill::zeros);
  m_fict_mass = arma::vec (m_nbead, arma::fill::zeros);

  m_rbath_nm  = arma::vec (m_nnhc,  arma::fill::zeros);
  m_vbath_nm  = arma::vec (m_nnhc,  arma::fill::zeros);
  m_qmass_nm  = arma::vec (m_nnhc,  arma::fill::zeros);

  // one-dimensional system 
  m_pos_qm    = arma::vec (m_nbead, arma::fill::zeros); 
  m_pos_nm    = arma::vec (m_nbead, arma::fill::zeros);
  
  m_vel_nm    = arma::vec (m_nbead, arma::fill::zeros);
  
  m_grd_qm    = arma::vec (m_nbead, arma::fill::zeros);
  m_grd_nm    = arma::vec (m_nbead, arma::fill::zeros);
  m_grd_nm_spr= arma::vec (m_nbead, arma::fill::zeros);


  // --- initiate the normal mode matrix.
  for (size_t i = 0; i < m_nbead; ++i) {
    m_tmat_nm(i, 0) = 1.0;
  }
    
  for (size_t i = 0; i < m_nbead/2; ++i) {
    m_tmat_nm(2*i,   m_nbead-1) = -1.0;
    m_tmat_nm(2*i+1, m_nbead-1) =  1.0;
  }

  double dnorm = sqrt (2.0);
    
  for (size_t i = 0; i < m_nbead; ++i) {
    const double di    = i+1;
    const double phase = 2.0*di*(M_PI/dbead);
    for (size_t j = 0; j < (m_nbead-2)/2; ++j) {
      const double dj    = j+1;
      m_tmat_nm(i, 2*j+1) = dnorm*cos(phase*dj);
      m_tmat_nm(i, 2*j+2) = dnorm*sin(phase*dj);
    }
  }

  
  // --- mass init ---
  double mass = m_mass;

  m_fict_mass(0)         = mass;
  m_fict_mass(m_nbead-1) = 4.0*dbead*mass;

  for (int ib = 1; ib < m_nbead/2; ++ib) {
    double val = 2.0*(1.0 - cos (2.0*ib*(M_PI/dbead)))*dbead*mass;
    m_fict_mass(2*ib-1) = val;
    m_fict_mass(2*ib  ) = val;
  }
  

  // bath init for beads
  
  m_qmass_nm(0) = m_gfree_nm*m_ekin_nm/m_omega_p2;
    
  for (size_t ihc = 1; ihc < m_nnhc; ++ihc) {
    m_qmass_nm(ihc) = m_ekin_nm/m_omega_p2;
  }


  nm_pos_init ();
  nm_vel_init ();
  
  prob_dx      = arma::vec(400, arma::fill::zeros);
  
}


void read_input (const std::string& fname)
{

  std::ifstream is_input(fname);
  std::ostringstream oss;
  oss << is_input.rdbuf();

  std::istringstream ss (oss.str());

  m_nstep = 1000;
  m_dt   = 0.5; // fsec
  m_irst = 0;
  
  m_nbead = 8;
  m_nref  = 10;
  m_ZPE   = 0.0; // kcal/mol

  m_mass  = 1.0*amu2au; // amu --> au
  
  std::string line;

  while(getline(ss, line)) {
    std::istringstream iss (line);
    std::string keyword;
    std::string val;
    iss >> keyword >> val;

    if (keyword == "nstep") {
      m_nstep = stoi (val);
    }
    else if (keyword == "dt") {
      m_dt = stod(val);
    }
    else if (keyword == "l_restart") {
      m_irst = stoi(val);
    }
    else if (keyword == "nbead") {
      m_nbead = stoi(val);
    }
    else if (keyword == "nref") {
      m_nref  = stoi(val);
    }
//    else if (keyword == "ZPE") {
//      m_ZPE  = stod(val)/au_kcal; // kcal/mol ---> au
//    }
    else if (keyword == "mass1") {
      m_mass = stod(val)*amu2au; // amu --> au
    }
    //else if (keyword == "mass2") {
    //  m_mass(1) = stod(val)*amu2au; // amu --> au
    //}
    
  }

  // -- time : [fs] ---> [au]
  m_dt  = m_dt * (1.0e-15/au_time);

  m_dt_ref = m_dt/m_nref;
  
}


} // namespace willow


int main (int argc, char *argv[])
{

  cout << std::setprecision (10);
  cout << std::fixed;


  //--- read an input file --
  const std::string fname = (argc > 1) ? argv[1] : "sample1.inp";

  willow::read_input (fname);

  willow::wpimd_init ();
  willow::wpimd_run ();

  return 0;
  
}
