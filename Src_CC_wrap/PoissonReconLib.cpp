// ##########################################################################
// #                                                                        #
// #               CLOUDCOMPARE WRAPPER: PoissonReconLib                    #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #               COPYRIGHT: Daniel Girardeau-Montaut                      #
// #                                                                        #
// ##########################################################################

// Local
#include "PoissonReconLib.h"

// Poisson (Header Only)
#include "../Src/PreProcessor.h"
#include "../Src/Reconstructors.h"

#include <cassert>
#include <thread>

using namespace PoissonRecon;

// The dimension of the system
static constexpr unsigned int DIMENSION = 3;

// Config
int PoissonReconLib::Parameters::GetMaxThreadCount()
{
	return static_cast<int>(std::thread::hardware_concurrency());
}

PoissonReconLib::Parameters::Parameters()
    : threads(GetMaxThreadCount())
{
}

// Poisson Color type helper. Color need to be composable.
template <typename Real>
struct RGBColor
{
	RGBColor(Real r = 0, Real g = 0, Real b = 0)
	    : r(r)
	    , g(g)
	    , b(b)
	{
	}

	RGBColor& operator+=(const RGBColor& c)
	{
		r += c.r;
		g += c.g;
		b += c.b;
		return *this;
	}
	RGBColor& operator*=(Real s)
	{
		r *= s;
		g *= s;
		b *= s;
		return *this;
	}
	RGBColor& operator/=(Real s)
	{
		return operator*=(1 / s);
	}

	RGBColor operator+(const RGBColor& c) const
	{
		return RGBColor(r + c.r, g + c.g, b + c.b);
	}
	RGBColor operator*(Real s) const
	{
		return RGBColor(r * s, g * s, b * s);
	}
	RGBColor operator/(Real s) const
	{
		return operator*(1 / s);
	}

	Real r, g, b;
};

// Atomic wrapper for color accumulation (thread safe)
namespace PoissonRecon
{
	template <typename Real>
	struct Atomic<RGBColor<Real>>
	{
		static void Add(volatile RGBColor<Real>& a, const RGBColor<Real>& b)
		{
			Atomic<Real>::Add(a.r, b.r);
			Atomic<Real>::Add(a.g, b.g);
			Atomic<Real>::Add(a.b, b.b);
		}
	};
} // namespace PoissonRecon

// Input
template <typename Real>
class PointStream : public Reconstructor::InputOrientedSampleStream<Real, DIMENSION>
{
  public:
	PointStream(const PoissonReconLib::ICloud<Real>& cloud)
	    : m_cloud(cloud)
	    , m_index(0)
	{
	}

	void reset() override
	{
		m_index = 0;
	}

	bool read(Point<Real, DIMENSION>& p, Point<Real, DIMENSION>& n) override
	{
		if (m_index >= m_cloud.size())
		{
			return false;
		}
		m_cloud.getPoint(m_index, p.coords);
		m_cloud.getNormal(m_index, n.coords);
		++m_index;
		return true;
	}

  protected:
	const PoissonReconLib::ICloud<Real>& m_cloud;
	size_t                               m_index;
};

template <typename Real>
class PointStreamWithColor : public Reconstructor::InputOrientedSampleStream<Real, DIMENSION, RGBColor<Real>>
{
  public:
	PointStreamWithColor(const PoissonReconLib::ICloud<Real>& cloud)
	    : m_cloud(cloud)
	    , m_index(0)
	{
	}

	void reset() override
	{
		m_index = 0;
	}

	bool read(Point<Real, DIMENSION>& p, Point<Real, DIMENSION>& n, RGBColor<Real>& color) override
	{
		if (m_index >= m_cloud.size())
		{
			return false;
		}
		m_cloud.getPoint(m_index, p.coords);
		m_cloud.getNormal(m_index, n.coords);
		float rgb[3];
		m_cloud.getColor(m_index, rgb);
		color = RGBColor<Real>(static_cast<Real>(rgb[0]), static_cast<Real>(rgb[1]), static_cast<Real>(rgb[2]));
		++m_index;
		return true;
	}

  protected:
	const PoissonReconLib::ICloud<Real>& m_cloud;
	size_t                               m_index;
};

template <typename Real>
class FaceStreamT : public Reconstructor::OutputFaceStream<DIMENSION - 1>
{
  public:
	FaceStreamT(PoissonReconLib::IMesh<Real>& mesh)
	    : m_mesh(mesh)
	{
	}
	size_t size() const override
	{
		return m_count;
	}
	size_t write(const std::vector<node_index_type>& polygon) override
	{
		assert(polygon.size() == 3);
		m_mesh.addTriangle(static_cast<size_t>(polygon[0]), static_cast<size_t>(polygon[1]), static_cast<size_t>(polygon[2]));
		return m_count++;
	}

  protected:
	PoissonReconLib::IMesh<Real>& m_mesh;
	size_t                        m_count = 0;
};

// Output

template <typename Real>
class VertexStream : public Reconstructor::OutputLevelSetVertexStream<Real, DIMENSION>
{
  public:
	VertexStream(PoissonReconLib::IMesh<Real>& mesh, bool outputDensity)
	    : m_mesh(mesh)
	    , m_outputDensity(outputDensity)
	{
	}

	size_t size() const override
	{
		return m_count;
	}

	size_t write(const Point<Real, DIMENSION>& p, const Point<Real, DIMENSION>& g, const Real& w) override
	{
		m_mesh.addVertex(p.coords);
		m_mesh.addNormal(g.coords);
		if (m_outputDensity)
		{
			m_mesh.addDensity(static_cast<double>(w));
		}
		return m_count++;
	}

  protected:
	PoissonReconLib::IMesh<Real>& m_mesh;
	bool                          m_outputDensity;
	size_t                        m_count = 0;
};

template <typename Real>
class VertexStreamWithColor : public Reconstructor::OutputLevelSetVertexStream<Real, DIMENSION, RGBColor<Real>>
{
  public:
	VertexStreamWithColor(PoissonReconLib::IMesh<Real>& mesh, bool outputDensity)
	    : m_mesh(mesh)
	    , m_outputDensity(outputDensity)
	{
	}

	size_t size() const override
	{
		return m_count;
	}

	size_t write(const Point<Real, DIMENSION>& p, const Point<Real, DIMENSION>& g, const Real& w, const RGBColor<Real>& c) override
	{
		m_mesh.addVertex(p.coords);
		m_mesh.addNormal(g.coords);
		float rgb[3] = {static_cast<float>(c.r), static_cast<float>(c.g), static_cast<float>(c.b)};
		m_mesh.addColor(rgb);
		if (m_outputDensity)
		{
			m_mesh.addDensity(static_cast<double>(w));
		}
		return m_count++;
	}

  protected:
	PoissonReconLib::IMesh<Real>& m_mesh;
	bool                          m_outputDensity;
	size_t                        m_count = 0;
};

template <typename Real, unsigned int FEMSig>
static bool ReconstructWithBoundary(const PoissonReconLib::ICloud<Real>& cloud,
                                    PoissonReconLib::IMesh<Real>&        outMesh,
                                    const PoissonReconLib::Parameters&   params,
                                    bool                                 withColor)
{
	using FEMSigs = IsotropicUIntPack<DIMENSION, FEMSig>;

	ThreadPool::SetNumTreads(params.threads);

	// Solver parameters
	using SolverParams = Reconstructor::Poisson::SolutionParameters<Real>;
	SolverParams solverParams;
	solverParams.depth                   = static_cast<unsigned int>(params.depth > 0 ? params.depth : 8);
	solverParams.width                   = static_cast<Real>(params.finestCellWidth);
	solverParams.scale                   = static_cast<Real>(params.scale);
	solverParams.samplesPerNode          = static_cast<Real>(params.samplesPerNode);
	solverParams.pointWeight             = static_cast<Real>(params.pointWeight);
	solverParams.iters                   = static_cast<unsigned int>(params.iters);
	solverParams.confidence              = params.confidence;
	solverParams.exactInterpolation      = params.exactInterpolation;
	solverParams.fullDepth               = static_cast<unsigned int>(params.fullDepth);
	solverParams.baseDepth               = static_cast<unsigned int>(params.baseDepth);
	solverParams.baseVCycles             = static_cast<unsigned int>(params.baseVCycles);
	solverParams.cgSolverAccuracy        = static_cast<Real>(params.cgAccuracy);
	solverParams.perLevelDataScaleFactor = static_cast<Real>(params.colorPullFactor);
	solverParams.alignDir                = 2; // Harcoded to match the Poisson CLI. But default to 0 is good as well

	// Extract parameters
	Reconstructor::LevelSetExtractionParameters extractionParams;
	extractionParams.linearFit       = params.linearFit; // better quality for noiseless points
	extractionParams.outputGradients = true;             // TODO(RJ:) output normals // we could make this optional.
	extractionParams.forceManifold   = true;
	extractionParams.gridCoordinates = false; // We always reconstruct mesh, do not export grid. TODO: could be a good idea to do alternative contouring (with CGAL for example)
	extractionParams.polygonMesh     = false;
	extractionParams.outputDensity   = params.density;

	FaceStreamT<Real> faceStream(outMesh);

	if (withColor)
	{
		using Implicit = Reconstructor::Implicit<Real, DIMENSION, FEMSigs, RGBColor<Real>>;
		using Solver   = Reconstructor::Poisson::Solver<Real, DIMENSION, FEMSigs, RGBColor<Real>>;

		PointStreamWithColor<Real> pointStream(cloud);
		std::unique_ptr<Implicit>  implicit(Solver::Solve(pointStream, solverParams, RGBColor<Real>()));
		if (!implicit)
		{
			return false;
		}

		VertexStreamWithColor<Real> vertexStream(outMesh, params.density);
// A race condition exists in Level set extraction (mkazhdan/PoissonRecon#190),
// on other arch than x86. Run extraction serially;
// the multigrid solve above stays parallel.
#if !defined(__x86_64__)
		ThreadPool::ParallelizationType = ThreadPool::ParallelType::NONE;
#endif
		implicit->extractLevelSet(vertexStream, faceStream, extractionParams);
	}
	else
	{
		using Implicit = Reconstructor::Implicit<Real, DIMENSION, FEMSigs>;
		using Solver   = Reconstructor::Poisson::Solver<Real, DIMENSION, FEMSigs>;

		PointStream<Real>         pointStream(cloud);
		std::unique_ptr<Implicit> implicit(Solver::Solve(pointStream, solverParams));
		if (!implicit)
		{
			return false;
		}

		VertexStream<Real> vertexStream(outMesh, params.density);

// see comment above
#if !defined(__x86_64__)
		ThreadPool::ParallelizationType = ThreadPool::ParallelType::NONE;
#endif
		implicit->extractLevelSet(vertexStream, faceStream, extractionParams);
	}

	return true;
}

template <typename Real>
bool PoissonReconLib::Reconstruct(const Parameters&   params,
                                  const ICloud<Real>& inCloud,
                                  IMesh<Real>&        outMesh)
{
	// unlikely to happen (should be tested beforehand)
	assert(inCloud.hasNormals());

#ifdef _OPENMP
	ThreadPool::ParallelizationType = ThreadPool::ParallelType::OPEN_MP;
#else
	// Use std::async
	ThreadPool::ParallelizationType = ThreadPool::ParallelType::ASYNC;
#endif
	try
	{
		// The default FEM degree for Poisson reconstruction / TODO is it worth it to make this configurable?
		static constexpr unsigned int DEFAULT_FEM_DEGREE = Reconstructor::Poisson::DefaultFEMDegree;

		bool useColor = params.withColors && inCloud.hasColors();

		// Boundary type Dispatcher. TODO: we can implement and dispatch to SSD here (see the example in upstream lib)
		switch (params.boundary)
		{
		case PoissonReconLib::Parameters::FREE:
			return ReconstructWithBoundary<Real, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_FREE>::Signature>(inCloud, outMesh, params, useColor);
		case PoissonReconLib::Parameters::DIRICHLET:
			return ReconstructWithBoundary<Real, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_DIRICHLET>::Signature>(inCloud, outMesh, params, useColor);
		case PoissonReconLib::Parameters::NEUMANN:
			return ReconstructWithBoundary<Real, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_NEUMANN>::Signature>(inCloud, outMesh, params, useColor);
		default:
			assert(false);
			return false;
		}
	}
	catch (...)
	{
		return false;
	}
}

// Explicit template instantiation
template bool PoissonReconLib::Reconstruct<float>(const Parameters&, const ICloud<float>&, IMesh<float>&);
template bool PoissonReconLib::Reconstruct<double>(const Parameters&, const ICloud<double>&, IMesh<double>&);
