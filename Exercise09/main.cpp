#include "Grid/grid_dist_id.hpp"
#include "data_type/aggregate.hpp"
#include "timer.hpp"

constexpr int U = 0;
constexpr int V = 1;

constexpr int x = 0;
constexpr int y = 1;


void init(
    grid_dist_id<2,double,aggregate<double,double> > & Old,
    grid_dist_id<2,double,aggregate<double,double> > & New,
    Box<2,double> & domain) 
{

	auto it = Old.getDomainIterator();

	while (it.isNext())
	{
		// Get the local grid key
		auto key = it.get();

		// Old values U and V
		Old.template get<U>(key) = 0.7;
		Old.template get<V>(key) = 0.04;

		// Old values U and V
		New.template get<U>(key) = 0.7;
		New.template get<V>(key) = 0.04;

		++it;
	}

	// grid_key_dx<2> start({(long int)std::floor(Old.size(0)*1.55f/domain.getHigh(0)),(long int)std::floor(Old.size(1)*1.55f/domain.getHigh(1))});
	// grid_key_dx<2> stop ({(long int)std::ceil (Old.size(0)*1.85f/domain.getHigh(0)),(long int)std::ceil (Old.size(1)*1.85f/domain.getHigh(1))});
	// auto it_init = Old.getSubDomainIterator(start,stop);

	// while (it_init.isNext())
	// {
	// 	auto key = it_init.get();

	// 	// Old.template get<U>(key) = 0.5 + (((double)std::rand())/RAND_MAX -0.5)/100.0;
	// 	// Old.template get<V>(key) = 0.25 + (((double)std::rand())/RAND_MAX -0.5)/200.0;

	// 	++it_init;
	// }

}


int main(int argc, char* argv[])
{

	openfpm_init(&argc,&argv);

	// domain
	Box<2,double> domain({0.0,0.0},{81.0, 81.0});
	
	// grid size
	size_t sz[2] = {51,51};

	// Define periodicity of the grid
	periodicity<2> bc = {PERIODIC,PERIODIC};
	
	// Ghost in grid unit
	Ghost<2,long int> g(1);
	
	// deltaT
	double deltaT = 0.01;

	// Diffusion constant for specie U
	double du = 2*1e-5;

	// Diffusion constant for specie V
	double dv = 1*1e-5;

	// Number of timesteps
	size_t timeSteps = 2000;

	// K and F (Physical constant in the equation)
	double K = 0.055;
	double F = 0.03;

	grid_dist_id<2, double, aggregate<double,double>> Old(sz,domain,g,bc);

	// New grid with the decomposition of the old grid
	grid_dist_id<2, double, aggregate<double,double>> New(Old.getDecomposition(),sz,g);

	
	// spacing of the grid on x and y
	double spacing[2] = {Old.spacing(0),Old.spacing(1)};

	init(Old,New,domain);

	// sync the ghost
	size_t count = 0;
	Old.template ghost_get<U,V>();

	// because we assume that spacing[x] == spacing[y] we use formula 2
	// and we calculate the prefactor of Eq 2
	double uFactor = deltaT * du/(spacing[x]*spacing[x]);
	double vFactor = deltaT * dv/(spacing[x]*spacing[x]);

    double a = 2.0;
    double b = 6.0;
    double k = 1.0;

	for (size_t i = 0; i < timeSteps; ++i)
	{
		auto it = Old.getDomainIterator();

		while (it.isNext())
		{
			auto key = it.get();

			// update based on Eq 2
			// New.get<U>(key) = Old.get<U>(key) + uFactor * (
			// 							Old.get<U>(key.move(x,1)) +
			// 							Old.get<U>(key.move(x,-1)) +
			// 							Old.get<U>(key.move(y,1)) +
			// 							Old.get<U>(key.move(y,-1)) +
			// 							-4.0*Old.get<U>(key)) +
			// 							- deltaT * Old.get<U>(key) * Old.get<V>(key) * Old.get<V>(key) +
			// 							- deltaT * F * (Old.get<U>(key) - 1.0);

            double gauss = sqrt(a);

            New.get<U>(key) = Old.get<U>(key) + deltaT*(
                a + k*Old.get<U>(key)*Old.get<U>(key)*Old.get<V>(key) - (b+1.0)*Old.get<U>(key)
            );

			// update based on Eq 2
			// New.get<V>(key) = Old.get<V>(key) + vFactor * (
			// 							Old.get<V>(key.move(x,1)) +
			// 							Old.get<V>(key.move(x,-1)) +
			// 							Old.get<V>(key.move(y,1)) +
			// 							Old.get<V>(key.move(y,-1)) -
			// 							4*Old.get<V>(key)) +
			// 							deltaT * Old.get<U>(key) * Old.get<V>(key) * Old.get<V>(key) +
			// 							- deltaT * (F+K) * Old.get<V>(key);

            New.get<V>(key) = Old.get<V>(key) + deltaT * (
                b*Old.get<U>(key) - k*Old.get<U>(key)*Old.get<U>(key)*Old.get<V>(key)
            );

			// Next point in the grid
			++it;
		}

		// Here we copy New into the old grid in preparation of the new step
		// It would be better to alternate, but using this we can show the usage
		// of the function copy. To note that copy work only on two grid of the same
		// decomposition. If you want to copy also the decomposition, or force to be
		// exactly the same, use Old = New
		Old.copy(New);

		// After copy we synchronize again the ghost part U and V
		Old.ghost_get<U,V>();

		// Every 100 time step we output the configuration for
		// visualization
		if (i % 10 == 0)
		{
			Old.ghost_get<U,V>();
			Old.write_frame("vtk_output/output",count);
			count++;
		}
	}

	openfpm_finalize();
}
