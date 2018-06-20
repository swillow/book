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

extern "C"
{
  void pot_nasa_ (const double* pos, double& en, double* grd);
}


namespace willow {

// global variables
const  int m_ndim  = 3;
const  int m_natom = 3;
const  int m_nnhc  = 4;
static int m_nbead;
static int m_nstep;
static int m_irst;
static int m_nref;
static double m_dt;
static double m_dt_ref;
static double m_gfree_nm;

static double m_ZPE;
static double m_omega;
static double m_omega_p2;
static double m_temp_nm;
static double m_beta_nm;
static double m_ekin_nm;

//const  double roh0   = 0.9586490;
const  double roh0   = 0.95784;
const  double delt_x = 2.e-3;

static arma::vec prob_roh;      // (0:399)

static arma::vec3 m_mass;     // (m_natom)

static arma::mat  m_tmat_nm;  // (m_nbead, m_nbead)
static arma::mat  m_fict_mass;// (m_natom, m_nbead)

static arma::vec  m_rbath_nm; // (m_nnhc)
static arma::vec  m_vbath_nm; //
static arma::vec  m_qmass_nm; //

// two-dimensional system
// Cartesian Coordinate
static double      m_e1_cent;
static arma::mat   m_pos_cent;
static arma::mat   m_grd_cent_A;

static arma::cube  m_pos_qm;   // (m_ndim, m_natom, m_nbead)
static arma::cube  m_grd_qm;   // (m_ndim, m_natom, m_nbead)

// Normal Mode
static arma::cube  m_pos_nm;   // (m_ndim, m_natom, m_nbead)
static arma::cube  m_vel_nm;   // (m_ndim, m_natom, m_nbead)
static arma::cube  m_grd_nm;   // (m_ndim, m_natom, m_nbead)
static arma::cube  m_grd_nm_spr; // (m_ndim, m_natom, m_nbead)
  

static void sample_rho ()
{

  // QM (beads)
  const size_t io  = 0;
  const size_t ih1 = 1;
  const size_t ih2 = 2;
  
  for (auto ib = 0; ib < m_nbead; ++ib) {
    
    // (1) \rho(x) = \rho (x2-x1)
    arma::mat pos = m_pos_qm.slice(ib)*bohr2ang;

    arma::vec3 ri_o  = pos.col(io);
    arma::vec3 ri_h1 = pos.col(ih1);
    arma::vec3 ri_h2 = pos.col(ih2);
    
    // OH1
    arma::vec3 dr = ri_o - ri_h1;
    double    dx  = arma::dot(dr, dr);
    int  id1  = (int) round( (dx-roh0) /delt_x) + 200;
    
    if (id1 >= 0 && id1 < 400) prob_roh(id1) += 1;
    
    dr  = ri_o - ri_h2; 
    dx  = arma::dot(dr, dr);
    id1  = (int) round( (dx-roh0) /delt_x) + 200;
    
    if (id1 >= 0 && id1 < 400) prob_roh(id1) += 1;
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
  
  double ekin_nm = 0.0;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {

    arma::mat vel = m_vel_nm.slice(ib);
    
    for (size_t ia = 0; ia < m_natom; ++ia) {

      double mass  = m_fict_mass (ia, ib);
      arma::vec  v = vel.col(ia);

      ekin_nm += mass * arma::dot (v, v);
      
    }

  }
  
  //---
    
  vbath = m_vbath_nm;
  rbath = m_rbath_nm;
  qmass = m_qmass_nm;

  // ihc = 0
  fbath(0) = (ekin_nm - m_gfree_nm*m_ekin_nm)/qmass(0);
    
  for (size_t ihc = 1; ihc < m_nnhc; ++ihc) {
    fbath(ihc) =
      (qmass(ihc-1)*vbath(ihc-1)*vbath(ihc-1) - m_ekin_nm) / qmass(ihc);
  }
  
  // Update Thermostat Velocities
  
  vbath(m_nnhc-1) = vbath(m_nnhc-1) + fbath(m_nnhc-1)*dt_ref4;
    
  for (auto ihc = 1; ihc < m_nnhc; ++ihc) {
    const auto jhc     = m_nnhc - ihc;
    const double vfact = exp (-vbath(jhc)*dt_ref8);
    const double vtmp  = vbath(jhc-1);
    vbath(jhc-1) = vtmp*vfact*vfact + fbath(jhc-1)*vfact*dt_ref4;
  }
  
  // Update atomic velocities
  const double pvfact = exp(-vbath(0)*dt_ref2);
  
  ekin_nm  = ekin_nm*pvfact*pvfact;
  
  // Update thermostat forces
  
  fbath(0) = (ekin_nm - m_gfree_nm*m_ekin_nm)/qmass(0);
    
  for (auto ihc = 0; ihc < m_nnhc; ++ihc) {
    rbath(ihc) += vbath(ihc)*dt_ref2;
  }
  
  // Update Thermostat velocities
    
  for (auto ihc = 0; ihc < m_nnhc-1; ++ihc) {
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
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    m_vel_nm.slice(ib) *= pvfact;
  }
  
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
  m_grd_nm_spr.slice(0).zeros();
  
  //
  // Gradients from the Springs between 'neighboring' beads
  //
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    
    for (size_t ia = 0; ia < m_natom; ++ia) {
    
      const double fact = m_fict_mass(ia,ib)*m_omega_p2;

      m_grd_nm_spr.slice(ib).col(ia) = fact * m_pos_nm.slice(ib).col(ia);
    }
  }
  
  
}





void nm_pos_trans ()
{
  //
  // normal mode (nm) ---> Cartesian (qm)
  // pos_nm --->  pos_qm
  
  for (auto ib = 0; ib < m_nbead; ++ib) {
    
    arma::mat pos_x (m_ndim, m_natom, arma::fill::zeros);
    
    for (auto jb = 0; jb < m_nbead; ++jb) {
      pos_x += m_tmat_nm(ib,jb)*m_pos_nm.slice(jb);
    }
    
    m_pos_qm.slice(ib) = pos_x;
  }

}




void nm_grad_trans ()
{

  m_grd_nm.zeros();
  
  for (size_t ib = 0; ib < m_nbead; ++ib) {
    for (size_t jb = 0; jb < m_nbead; ++jb) {
      m_grd_nm.slice(ib) += m_tmat_nm(jb,ib)*m_grd_qm.slice(jb);
    }
  }

}




void nm_pos_init () 
{

  {// centroid particle
    m_pos_nm.slice(0) = m_pos_cent;
  }

  { // beads : normal mode
    const double dbead  = m_nbead;
    const double usigma = 0.02*ang2bohr; // sigma_x = 0.02 A
    
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {
      
      arma::mat pos_nm_ib(m_ndim, m_natom, arma::fill::zeros);
      
      for (size_t ia = 0; ia < m_natom; ++ia) {

	double inv_mass05 = 1.0/sqrt(m_fict_mass(ia,ib));
	pos_nm_ib(0,ia) = usigma*rnd::gaus_dev()*inv_mass05;
	pos_nm_ib(1,ia) = usigma*rnd::gaus_dev()*inv_mass05;
	pos_nm_ib(2,ia) = usigma*rnd::gaus_dev()*inv_mass05;
      }

      m_pos_nm.slice(ib) = pos_nm_ib;
      
    }
  } // beads

}


void nm_vel_init ()
{

  // ---- centroid ----
  {
    // Here, vel_nm(ib = 0) zero
    m_vel_nm.slice(0).zeros();
  }
  
  { // velocities for bead particles (or nm-mode particles)

    arma::vec3 v;
    arma::vec3 sump;
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {
      
      sump.zeros();
      double sum_mass = 0.0;
      
      for (size_t ia = 0; ia < m_natom; ++ia) {
	
	double mass   = m_fict_mass(ia,ib);
	double vsigma = sqrt (m_ekin_nm/mass);

	v(0) = vsigma*rnd::gaus_dev();
	v(1) = vsigma*rnd::gaus_dev();
	v(2) = vsigma*rnd::gaus_dev();

	m_vel_nm.slice(ib).col(ia) = v; 
	
	sump     += mass*v;
	sum_mass += mass;
      }
      
      sump /= sum_mass;

      // translational motion is zero
      for (auto ia = 0; ia < m_natom; ++ia) {
	m_vel_nm.slice(ib).col(ia) -= sump;
      }
      
    }

    //
    // subtract rotation
    //
    for (size_t ib = 1; ib < m_nbead; ++ib) {

      arma::vec3 pos_com;
      pos_com.zeros();

      double sum_mass = 0.0;

      for (size_t ia = 0; ia < m_natom; ++ia) {

	double mass = m_fict_mass(ia,ib);
	arma::vec3 q3 = m_pos_nm.slice(ib).col(ia);

	pos_com += mass * q3;
	sum_mass += mass;
      }

      pos_com /= sum_mass;

      // angular momentum

      arma::vec3 bv;
      bv.zeros();

      for (size_t ia = 0; ia < m_natom; ++ia) {
	double mass = m_fict_mass(ia,ib);
	arma::vec3 r3 = m_pos_nm.slice(ib).col(ia) - pos_com;
	arma::vec3 v3 = m_vel_nm.slice(ib).col(ia);

	bv += mass * arma::cross (r3, v3);
      }
      

      // moment of inertia

      arma::mat am(3,3,arma::fill::zeros);

      for (size_t ia = 0; ia < m_natom; ++ia) {
	double mass = m_fict_mass(ia,ib);
	arma::vec3 r3 = m_pos_nm.slice(ib).col(ia) - pos_com;
	  
	am(0,0) += mass*(r3(1)*r3(1)  + r3(2)*r3(2)); // (y*y + z*z)
	am(0,1) -= mass*r3(0)*r3(1); // x*y
	am(0,2) -= mass*r3(0)*r3(2); // x*z
	
	//am(1,0) -= mass*r3(1)*r3(0); // y*x
	am(1,1) += mass*(r3(2)*r3(2) + r3(0)*r3(0)); // z*z + x*x
	am(1,2) -= mass*r3(1)*r3(2); // y*z
	
	//am(2,0) -= mass*r3(2)*r3(0); // z*x;
	//am(2,1) -= mass*r3(2)*r3(1); // z*y;
	am(2,2) += mass*(r3(0)*r3(0) + r3(1)*r3(1)); // x*x + y*y
      }
	
      am(1,0) = am(0,1);
      am(2,0) = am(0,2);
      am(2,1) = am(1,2);
	
      // principal axis: diagonalize moment of interia
      arma::vec3 eg_val;
      arma::mat  eg_vec (3,3,arma::fill::zeros);

      arma::eig_sym (eg_val, eg_vec, am);

      // in principal axis: angular momentum
      // --- check
      arma::vec3 dv = eg_vec.t()*bv;

      dv(0) = dv(0)/eg_val(0);
      dv(1) = dv(1)/eg_val(1);
      dv(2) = dv(2)/eg_val(2);

      // d in laboratory frame

      arma::vec3 fv3 = eg_vec*dv;

      for (size_t ia = 0; ia < m_natom; ++ia) {
	arma::vec3 rv3 = m_pos_nm.slice(ib).col(ia) - pos_com;
	m_vel_nm.slice(ib).col(ia) -= arma::cross (fv3, rv3);
      }

    } // ib

    
    //
    // Scale Velocity 
    double ekin_nm = 0.0;
    
    for (size_t ib = 1; ib < m_nbead; ++ib) {
	
      for (size_t ia = 0; ia < m_natom; ++ia) {
	
	double mass  = m_fict_mass(ia,ib);
	v     = m_vel_nm.slice(ib).col(ia);
	ekin_nm += mass * arma::dot(v, v); 
	
      }
      
    }
    
    double temp  = ekin_nm / (m_gfree_nm*boltz);
    double scale = sqrt(m_temp_nm / temp);

    for (size_t ib = 1; ib < m_nbead; ++ib) {

      m_vel_nm.slice(ib) *= scale;
	
    } // ib
    
  } // velocities for bead particles

}



void nm_vel_update ()
{

  // (ib = 0) belongs to the centroid velocity

  
  //---
  const double dt2 = 0.5*m_dt;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    for (size_t ia = 0; ia < m_natom; ++ia) {
      double mass   = m_fict_mass(ia,ib); 
      double factor = dt2/mass;
      m_vel_nm.slice(ib).col(ia) -= factor*m_grd_nm.slice(ib).col(ia);
    }
  }

}




void nm_vel_spring_update ()
{

  const double dt2 = 0.5*m_dt_ref;
  
  for (size_t ib = 1; ib < m_nbead; ++ib) {
    
    for (size_t ia = 0; ia < m_natom; ++ia) {
      const double mass   = m_fict_mass(ia,ib); 
      const double factor = dt2/mass;
      m_vel_nm.slice(ib).col(ia) -=
	factor*m_grd_nm_spr.slice(ib).col(ia);
    }

  }

}


double nm_pot_spring ()
{
  // Harmonic Potential of springs between neighboring beads

  double qkin_nm = 0.0;
  for (auto ib = 1; ib < m_nbead; ++ib)
    for (auto ia = 0; ia < m_natom; ++ia) {
      const double fact  = 0.5*m_fict_mass(ia,ib)*m_omega_p2;
      arma::vec3   q3    = m_pos_nm.slice(ib).col(ia);
      qkin_nm += fact*arma::dot(q3,q3);
    }
  
  return qkin_nm;
}



double nm_pot_grad ()
{
  
  // beads: normal mode ---> cartesian
  //        pos_nm ----> pos_qm
  nm_pos_trans ();

  arma::cube pos_qm_A = m_pos_qm*bohr2ang;
  arma::mat  grd_qm_A (m_ndim, m_natom, arma::fill::zeros);

  double u_vib = 0.0;
  m_grd_qm.zeros();

  //
  for (size_t ib = 0; ib < m_nbead; ib++) {
    
    // call your potential.
    double e1;
    pot_nasa_ (pos_qm_A.slice(ib).memptr(), e1, grd_qm_A.memptr());

    // kcal/mol --> au
    u_vib += (e1 - m_e1_cent)/au_kcal;

    // kcal/mol/A ---> au/bohr
    m_grd_qm.slice(ib) = (grd_qm_A - m_grd_cent_A)/au_kcal/ang2bohr;
    
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
		const double& qkin_nm)
{
  // kinetic energy for beads
  double ekin_nm = 0.0;
  
  for (auto ib = 1; ib < m_nbead; ++ib) {
    for (auto ia = 0; ia < m_natom; ++ia) {
      const double mass = m_fict_mass(ia,ib);
      arma::vec3 v3     = m_vel_nm.slice(ib).col(ia); 
      ekin_nm += mass*arma::dot(v3, v3);
    }
  }

  ekin_nm = 0.5*ekin_nm;
  double temp_nm = 2.0*ekin_nm/(m_gfree_nm*boltz);

  double ebath_nm = 0.0;

  arma::vec4 qmass = m_qmass_nm;
  ebath_nm += 0.5*qmass(0)*m_vbath_nm(0)*m_vbath_nm(0);
  ebath_nm += m_gfree_nm*m_ekin_nm*m_rbath_nm(0);

  for (auto ihc = 1; ihc < m_nnhc; ++ihc) {
    ebath_nm += 0.5*qmass(ihc)*m_vbath_nm(ihc)*m_vbath_nm(ihc);
    ebath_nm += m_ekin_nm*m_rbath_nm(ihc);
  }
  

  double E_eff = qkin_nm + u_vib; // <E_eff> = E_ZPE
  double H_sys = ekin_nm + E_eff; // Hamiltonian of the system
  double H_tot = H_sys + ebath_nm; // Total H.

  // unit convert
  E_eff *= au_kcal;
  H_sys *= au_kcal;
  H_tot *= au_kcal;

  printf (" %8d %14.6f %14.6f %14.6f %14.6f %14.6f %10.2f \n",
	  (istep+1), H_tot, H_sys, m_e1_cent + E_eff, E_eff, 
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

  arma::vec3 v3;
  // ib = 0 is for centroids
  for (auto ib = 1; ib < m_nbead; ++ib) {
    for (auto ia = 0; ia < m_natom; ++ia) {
      std::getline (is, line);
      std::istringstream iss_pos (line);
    
      iss_pos >> v3(0) >> v3(1) >> v3(2);
      m_pos_nm.slice(ib).col(ia) = v3;
    }
  }
  
  for (auto ib = 1; ib < m_nbead; ++ib) {
    for (auto ia = 0; ia < m_natom; ++ia) {
      std::getline (is, line);
      std::istringstream iss_vel (line);

      iss_vel >> v3(0) >> v3(1) >> v3(2);
      m_vel_nm.slice(ib).col(ia) = v3;
    }
  }

  for (auto ihc = 0; ihc < m_nnhc; ++ihc) {
    std::getline (is,line);
    std::istringstream iss (line);

    iss >> m_rbath_nm(ihc) >> m_vbath_nm(ihc);
  }

  m_pos_nm.slice(0) = m_pos_cent;
  m_vel_nm.slice(0).zeros(); 
  m_grd_nm.slice(0).zeros(); 
  
}


void write_restart_nm (const int& istep)
{

  FILE *ofs_rst = fopen ("pimdrr.sav", "w");

  fprintf( ofs_rst, " %10d \n", istep+1);

  // ib = 0 is for centroids
  arma::vec3 v3;
  
  for (auto ib = 1; ib < m_nbead; ++ib) {
    for (auto ia = 0; ia < m_natom; ++ia) {
      v3 = m_pos_nm.slice(ib).col(ia);
      fprintf (ofs_rst, "  %E   %E   %E  \n", v3(0), v3(1), v3(2));
    }
  }
  
  for (auto ib = 1; ib < m_nbead; ++ib) {
    for (auto ia = 0; ia < m_natom; ++ia) {
      v3 = m_vel_nm.slice(ib).col(ia);
      fprintf (ofs_rst, "  %E   %E   %E  \n", v3(0), v3(1), v3(2));
    }
  }

  for (auto ihc = 0; ihc < m_nnhc; ++ihc) {
    fprintf (ofs_rst, "  %E   %E   \n",
	     m_rbath_nm(ihc),
	     m_vbath_nm(ihc) );
//	     m_qmass_nm(ihc) );
  }
  
  fclose (ofs_rst);

}


void write_prob_bin (int& nsamp)
{

  arma::vec p_dx = prob_roh/(2.0*m_nbead*(nsamp+1)*delt_x);
  
  std::ofstream ofs_qm ("prob_bin_qm.dat");
  
  for (auto ib = 0; ib < 400; ++ib) {
    double  x  = (ib - 200)*delt_x + roh0;
    ofs_qm << x << "   "  << p_dx(ib) << endl;
  }

  ofs_qm.close();

}


void wpimd_run ()
{

  //
  // This is a sample code,
  // in which WPIMD is running at T (thermal temperature) = 0 K.
  //
  FILE* ofs_report = fopen ("wpimd.report", "w");
  FILE* ofs_bead   = fopen ("wpimd.bead", "w");
  
  m_e1_cent = 0.0;
  m_grd_cent_A.zeros();
  {
    arma::mat pos_cent_A = m_pos_cent*bohr2ang;
    pot_nasa_ (pos_cent_A.memptr(), m_e1_cent, m_grd_cent_A.memptr());
  }

  m_pos_nm.slice(0) = m_pos_cent;
  
  int istep0 = 0;

  if (m_irst == 1) {
    read_restart_nm (istep0);
  }
  
  // initial gradients and potential energy
  double u_vib   = nm_pot_grad ();
  double qkin_nm = nm_pot_spring ();
  
  double eff_ave   = (u_vib + qkin_nm)*au_kcal;
  long double eff_var   = 0.0;
  
  nm_grad_spring ();

  //if (m_irst == 0)
  //  nm_report (-1, u_vib, qkin_nm);

  for (auto istep = 0; istep < m_nstep; ++istep) {

    nm_vel_update ();

    for (auto iref = 0; iref < m_nref; ++iref) {
      nm_nhc_integrate ();
      nm_vel_spring_update();
      nm_pos_update ();
      nm_grad_spring();
      nm_vel_spring_update();
      nm_nhc_integrate ();
    }

    u_vib   = nm_pot_grad ();
    qkin_nm = nm_pot_spring ();
    
    double eff  = (u_vib + qkin_nm)*au_kcal;
    double diff = eff - eff_ave;
    
    eff_ave += diff/(istep+1);
    eff_var += diff*diff*((double)istep)/(istep+1);

    if (abs(u_vib + qkin_nm - m_ZPE) < 1.0e-5) {
//    arma::mat grd_qm = m_grd_nm.slice(0)*au_kcal*ang2bohr;
//    if (abs(u_vib + qkin_nm - m_ZPE) < 2.0e-5) {
      // store
      fprintf (ofs_bead, " BEAD \n");
      arma::vec3 v3;
      for (auto ib = 1; ib < m_nbead; ++ib) {
	for (auto ia = 0; ia < m_natom; ++ia) {
	  v3 = m_pos_nm.slice(ib).col(ia);
	  fprintf (ofs_bead, " %E  %E  %E \n",
		   v3(0), v3(1), v3(2));
	}
      }
      fflush (ofs_bead);
    }
    
    sample_rho ();
    nm_vel_update ();
    
    if ( (istep+1)%1000 == 0) {
      nm_report (istep, u_vib, qkin_nm);
      double time_ps = (istep+1)*m_dt*au_time*1.0e12;
      double std = sqrt(eff_var/((double)istep*(istep+1)));
      fprintf (ofs_report, " %14.3f  %14.6f %14.6f %14.6f \n",
	       time_ps, m_e1_cent + eff_ave, eff_ave, std);
      fflush (ofs_report);
    }
    
    if ( (istep+1)%5000 == 0) {
      write_restart_nm   (istep+istep0);
      write_prob_bin (istep);
    }
    
  }

  fclose (ofs_report);
  fclose (ofs_bead);
  
  cout << "AVE E_eff " << eff_ave << endl;
  cout << "AVE <E>   " << m_e1_cent + eff_ave << endl;

  
}




void wpimd_init ()
{

  // two-dimensional system
  // 3 = 3*3 - 6
  m_gfree_nm = 3*m_nbead; // the vibrational degree of freedom
  
  // ZPE = 0.5 * hbar * omega
  // ZPE(au) = 0.5 * omega
  m_omega    = 2.0*m_ZPE;

  // Eq. (14)
  // temperature for the bead motions
  m_temp_nm  = m_omega/(m_gfree_nm*boltz);

  double dbead = m_nbead;
  double omega_p = sqrt(dbead)*boltz*m_temp_nm;
  m_omega_p2 = omega_p * omega_p;
  m_beta_nm  = 1.0 / (boltz*m_temp_nm);
  m_ekin_nm  = boltz*m_temp_nm;
  
  // mem alloc

  m_tmat_nm   = arma::mat (m_nbead, m_nbead, arma::fill::zeros);
  m_fict_mass = arma::mat (m_natom, m_nbead, arma::fill::zeros);

  m_rbath_nm  = arma::vec (m_nnhc, arma::fill::zeros);
  m_vbath_nm  = arma::vec (m_nnhc, arma::fill::zeros);
  m_qmass_nm  = arma::vec (m_nnhc, arma::fill::zeros);

  // one-dimensional system 
  m_grd_cent_A= arma::mat  (m_ndim, m_natom, arma::fill::zeros); 
  m_pos_qm    = arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros); 
  m_pos_nm    = arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros);
  
  m_vel_nm    = arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros);
  
  m_grd_qm    = arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros);
  m_grd_nm    = arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros);
  m_grd_nm_spr= arma::cube (m_ndim, m_natom, m_nbead, arma::fill::zeros);

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
  for (auto ia = 0; ia < m_natom; ++ia) {
    double mass = m_mass(ia);

    m_fict_mass(ia,0)         = mass;
    m_fict_mass(ia,m_nbead-1) = 4.0*dbead*mass;

    for (auto ib = 1; ib < m_nbead/2; ++ib) {
      double val = 2.0*(1.0 - cos (2.0*ib*(M_PI/dbead)))*dbead*mass;
      m_fict_mass(ia,2*ib-1) = val;
      m_fict_mass(ia,2*ib  ) = val;
    }
  }

  // bath init for beads
  
    
  m_qmass_nm(0) = m_gfree_nm*m_ekin_nm/m_omega_p2;
    
  for (size_t ihc = 1; ihc < m_nnhc; ++ihc) {
    m_qmass_nm(ihc) = m_ekin_nm/m_omega_p2;
  }
  

  nm_pos_init ();
  nm_vel_init ();
  
  prob_roh      = arma::vec(400, arma::fill::zeros);
  
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

  for (auto ia = 0; ia < m_natom; ++ia)
    m_mass(ia)  = 1.0*amu2au; // amu --> au
  
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
    else if (keyword == "ZPE") {
      m_ZPE  = stod(val)/au_kcal; // kcal/mol ---> au
    }
    else if (keyword == "mass1") {
      m_mass(0) = stod(val)*amu2au; // amu --> au
    }
    else if (keyword == "mass2") {
      m_mass(1) = stod(val)*amu2au; // amu --> au
    }
    else if (keyword == "mass3") {
      m_mass(2) = stod(val)*amu2au; // amu --> au
    }
    
  }

  // -- time : [fs] ---> [au]
  m_dt  = m_dt * (1.0e-15/au_time);

  m_dt_ref = m_dt/m_nref;

}

void read_geom (const std::string& fname)
{

  m_pos_cent = arma::mat (m_ndim, m_natom, arma::fill::zeros);
  
  std::ifstream is_geom (fname);
  std::ostringstream oss;
  oss << is_geom.rdbuf();

  std::istringstream ss (oss.str());
  std::string line;
  { // natom
    getline (ss, line);
    // comment
    getline (ss, line);
  }
  
  for (auto ia = 0; ia < m_natom; ++ia) {
    getline (ss, line);
    std::istringstream iss (line);
    std::string atnm;
    arma::vec3  v3;
    iss >> atnm >> v3(0) >> v3(1) >> v3(2);

    m_pos_cent.col(ia) = v3*ang2bohr;
  }
  
}



} // namespace willow


int main (int argc, char *argv[])
{

  cout << std::setprecision (6);
  cout << std::fixed;


  //--- read an input file --
  const std::string fname = (argc > 1) ? argv[1] : "sample_ps.inp";
  const std::string fname_geom = "geom.xyz";
  
  willow::read_input (fname);
  willow::read_geom  (fname_geom);
  willow::wpimd_init ();
  willow::wpimd_run ();

  return 0;
  
}
