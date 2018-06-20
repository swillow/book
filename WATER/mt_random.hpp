#ifndef W_MT_RANDOM_HPP
#define W_MT_RANDOM_HPP

#include <random>
#include <cmath>

namespace willow { namespace rnd {

static std::mt19937    mt;
static std::uniform_real_distribution<double> dist (0.0, 1.0);

inline void random_init (unsigned int iseed) {
    
    std::mt19937 gen (iseed);

    mt = gen;
}


inline double randf ()
{
  return dist(mt);
}


inline double gaus_dev ()
{
    // generate normal distribution (-1:1)

    static double surplus;
    static bool   surplus_ready (false);

    if (surplus_ready) {
	surplus_ready = false;

	return surplus;
    }
    else {
	double x1, x2, w;

	do {
	    x1 = 2.0* dist(mt) - 1.0;
	    x2 = 2.0* dist(mt) - 1.0;
	    w  = x1*x1 + x2*x2;
	} while (w >= 1.0);

	const double fac = std::sqrt ( (-2.0*std::log(w))/w );

	surplus = x1*fac;
	surplus_ready = true;

	return x2*fac;
    }


}


} } // namespace willow::rnd


#endif
