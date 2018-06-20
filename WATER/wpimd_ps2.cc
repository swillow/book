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
static int m_nbead;

static double m_roh0;
static double m_theta0;

static arma::mat  m_tmat_nm;  // (m_nbead, m_nbead)

void wpimd_run ()
{

  //
  // This is a sample code,
  // in which WPIMD is running at T (thermal temperature) = 0 K.
  //
  std::ifstream ofs_bead("wpimd.bead");
  std::ostringstream oss;
  oss << ofs_bead.rdbuf();
  std::istringstream ss (oss.str());
  
  ofs_bead.close();

  //const double r0     = 0.957840;
  //const double theta0 = 104.5080;
  const double dr     = 0.001;
  const double radian = 57.29577951308232088;
  
  double rang = m_theta0/radian;
  double cn   = std::cos(0.5*rang);
  double sn   = std::sin(0.5*rang);

  std::vector<arma::mat> pos_nm_o;
  std::vector<arma::mat> pos_nm_h1;
  std::vector<arma::mat> pos_nm_h2;

  std::string line;
  while (std::getline(ss, line)) {

    arma::vec3 v3;
    arma::mat  o_nm (3,m_nbead, arma::fill::zeros);
    arma::mat h1_nm (3,m_nbead, arma::fill::zeros);
    arma::mat h2_nm (3,m_nbead, arma::fill::zeros);

    for (auto ib = 1; ib < m_nbead; ++ib) {
      std::getline(ss, line);
      std::istringstream iss_o(line);
      iss_o >> v3(0) >> v3(1) >> v3(2);
      o_nm.col(ib) = v3;
      
      std::getline(ss, line);
      std::istringstream iss_h1(line);
      iss_h1 >> v3(0) >> v3(1) >> v3(2);
      h1_nm.col(ib) = v3;
      
      std::getline(ss, line);
      std::istringstream iss_h2(line);
      iss_h2 >> v3(0) >> v3(1) >> v3(2);
      h2_nm.col(ib) = v3;
    }

    pos_nm_o.push_back  (o_nm);
    pos_nm_h1.push_back (h1_nm);
    pos_nm_h2.push_back (h2_nm);
    
  }

  const int nsamp = pos_nm_o.size();

  arma::mat p_cm_A(3, 3, arma::fill::zeros);
  arma::mat g_cm_A(3, 3, arma::fill::zeros);
  double    e1_cm = 0.0;
  
  for (auto iconf = 0; iconf < 25; ++iconf) {
    double r1 = iconf*dr + (m_roh0 - 0.01);

    // O
    p_cm_A(0,0) = 0.0;
    p_cm_A(1,0) = 0.0;
    p_cm_A(2,0) = 0.0;
    // H1 
    p_cm_A(0,1) =  r1*sn;
    p_cm_A(1,1) = -r1*cn;
    p_cm_A(2,1) = 0.0;
    // H2 
    p_cm_A(0,2) = -r1*sn;
    p_cm_A(1,2) = -r1*cn;
    p_cm_A(2,2) = 0.0;

    e1_cm = 0.0;
    
    {
      pot_nasa_ (p_cm_A.memptr(), e1_cm, g_cm_A.memptr());
    }

    arma::mat p_cm = p_cm_A*ang2bohr;

    double u_vib = 0.0;
    
    for (auto is = 0; is < nsamp; ++is) {

      arma::mat nm_o  = pos_nm_o[is];
      arma::mat nm_h1 = pos_nm_h1[is];
      arma::mat nm_h2 = pos_nm_h2[is];
      
      nm_o.col(0)  = p_cm.col(0);
      nm_h1.col(0) = p_cm.col(1);
      nm_h2.col(0) = p_cm.col(2);
      
      for (size_t ib = 0; ib < m_nbead; ++ib) {
	
	arma::mat p_qm (3, 3, arma::fill::zeros);
	
	// nm --> qm
	for (size_t jb = 0; jb < m_nbead; ++jb) {
	  p_qm.col(0) += m_tmat_nm(ib,jb)*nm_o.col(jb);
	  p_qm.col(1) += m_tmat_nm(ib,jb)*nm_h1.col(jb);
	  p_qm.col(2) += m_tmat_nm(ib,jb)*nm_h2.col(jb);
	}
      
	arma::mat p_qm_A = p_qm*bohr2ang;
	arma::mat g_qm_A (3, 3, arma::fill::zeros);
	double e1_qm = 0.0;
      
	pot_nasa_ (p_qm_A.memptr(), e1_qm, g_qm_A.memptr());
      
	u_vib += (e1_qm - e1_cm);

      } // ib
      
    } // for is 
    
    u_vib /= (nsamp*m_nbead);

    std::cout << "   "  << r1
	      << "   "  << e1_cm
	      << "   "  << u_vib
	      << "   "  << e1_cm + u_vib << std::endl;

    
  }
  
  return;
  
}




void wpimd_init ()
{

  double dbead = m_nbead;

  m_tmat_nm   = arma::mat (m_nbead, m_nbead, arma::fill::zeros);

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
  
}


void read_input (const std::string& fname)
{

  std::ifstream is_input(fname);
  std::ostringstream oss;
  oss << is_input.rdbuf();

  std::istringstream ss (oss.str());

  m_nbead = 8;
  m_roh0  = 0.95784;
  m_theta0= 104.5080;
  
  std::string line;

  while(getline(ss, line)) {
    std::istringstream iss (line);
    std::string keyword;
    std::string val;
    iss >> keyword >> val;

    if (keyword == "nbead") {
      m_nbead = stoi(val);
    }
    else if (keyword == "roh0") {
      m_roh0  = stod(val); // A
    }
    else if (keyword == "theta0") {
      m_theta0  = stod(val); // degree
    }
  }

}



} // namespace willow


int main (int argc, char *argv[])
{

  cout << std::setprecision (8);
  cout << std::fixed;

  //--- read an input file --
  const std::string fname = (argc > 1) ? argv[1] : "sample_ps2.inp";
  
  willow::read_input (fname);
  willow::wpimd_init ();
  willow::wpimd_run ();

  return 0;
  
}
