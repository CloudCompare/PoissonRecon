//##########################################################################
//#                                                                        #
//#               CLOUDCOMPARE WRAPPER: PoissonReconLib                    #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#               COPYRIGHT: Daniel Girardeau-Montaut                      #
//#                                                                        #
//##########################################################################

#include "PoissonReconLib.h"

//PoissonRecon
#include "../Src/FEMTree.h"
#include "../Src/VertexFactory.h"
#include "../Src/PreProcessor.h"

template< typename Real, unsigned int Dim, typename AuxData >
using InputOrientedPointStreamInfo = typename FEMTreeInitializer< Dim, Real >::template InputPointStream< VectorTypeUnion< Real, typename VertexFactory::NormalFactory< Real, Dim >::VertexType, AuxData > >;

template< typename Real, unsigned int Dim, typename AuxData >
using InputOrientedPointStream = typename InputOrientedPointStreamInfo< Real, Dim, AuxData >::StreamType;

template< class Real, unsigned int Dim >
XForm< Real, Dim + 1 > GetBoundingBoxXForm(Point< Real, Dim > min, Point< Real, Dim > max, Real scaleFactor)
{
	Point< Real, Dim > center = (max + min) / 2;
	Real scale = max[0] - min[0];
	for (int d = 1; d < Dim; d++) scale = std::max< Real >(scale, max[d] - min[d]);
	scale *= scaleFactor;
	for (int i = 0; i < Dim; i++) center[i] -= scale / 2;
	XForm< Real, Dim + 1 > tXForm = XForm< Real, Dim + 1 >::Identity(), sXForm = XForm< Real, Dim + 1 >::Identity();
	for (int i = 0; i < Dim; i++) sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
	unsigned int maxDim = 0;
	for (int i = 1; i < Dim; i++) if ((max[i] - min[i]) > (max[maxDim] - min[maxDim])) maxDim = i;
	XForm< Real, Dim + 1 > rXForm;
	for (int i = 0; i < Dim; i++) rXForm((maxDim + i) % Dim, (Dim - 1 + i) % Dim) = 1;
	rXForm(Dim, Dim) = 1;
	return rXForm * sXForm * tXForm;
}

template< class Real, unsigned int Dim, typename AuxData >
XForm< Real, Dim + 1 > GetPointXForm(InputOrientedPointStream< Real, Dim, AuxData > &stream, typename InputOrientedPointStreamInfo< Real, Dim, AuxData >::DataType d, Real scaleFactor)
{
	Point< Real, Dim > min, max;
	InputOrientedPointStreamInfo< Real, Dim, AuxData >::BoundingBox(stream, d, min, max);
	return GetBoundingBoxXForm(min, max, scaleFactor);
}

template< unsigned int Dim, typename Real >
struct ConstraintDual
{
	Real target, weight;
	ConstraintDual(void) : target(0), weight(0) {}
	ConstraintDual(Real t, Real w) : target(t), weight(w) {}
	CumulativeDerivativeValues< Real, Dim, 0 > operator()(const Point< Real, Dim >& p) const { return CumulativeDerivativeValues< Real, Dim, 0 >(target*weight); };
};

template< unsigned int Dim, typename Real >
struct SystemDual
{
	Real weight;
	SystemDual(void) : weight(0) {}
	SystemDual(Real w) : weight(w) {}
	//CumulativeDerivativeValues< Real, Dim, 0 > operator()(const Point< Real, Dim >& p, const CumulativeDerivativeValues< Real, Dim, 0 >& dValues) const { return dValues * weight; };
	CumulativeDerivativeValues< double, Dim, 0 > operator()(const Point< Real, Dim >& p, const CumulativeDerivativeValues< double, Dim, 0 >& dValues) const { return dValues * weight; };
};

//System
#include <cassert>

namespace {
	// The order of the B-Spline used to splat in data for color interpolation
	constexpr int DATA_DEGREE = 0;
	// The order of the B-Spline used to splat in the weights for density estimation
	constexpr int WEIGHT_DEGREE = 2;
	// The order of the B-Spline used to splat in the normals for constructing the Laplacian constraints
	constexpr int NORMAL_DEGREE = 2;
	// The default finite-element degree
	constexpr int DEFAULT_FEM_DEGREE = 1;
	// The dimension of the system
	constexpr int DIMENSION = 3;

	inline float ComputeNorm(const float vec[3])
	{
		return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	}
	
	inline double ComputeNorm(const double vec[3])
	{
		return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	}
}

int PoissonReconLib::Parameters::GetMaxThreadCount()
{
#ifdef WITH_OPENMP
	return omp_get_num_procs();
#else
	return std::thread::hardware_concurrency();
#endif
}

PoissonReconLib::Parameters::Parameters()
	: threads(GetMaxThreadCount())
{
}

template< typename VertexFactory, typename Real, typename SetVertexFunction, typename InputSampleDataType, unsigned int ... FEMSigs >
StreamingMesh< typename VertexFactory::VertexType, node_index_type >* ExtractMesh
(
	const PoissonReconLib::Parameters& params,
	UIntPack< FEMSigs ... >,
	FEMTree< sizeof ... (FEMSigs), Real >& tree,
	const DenseNodeData< Real, UIntPack< FEMSigs ... > >& solution,
	Real isoValue,
	const std::vector< typename FEMTree< sizeof ... (FEMSigs), Real >::PointSample >* samples,
	std::vector< InputSampleDataType >* sampleData,
	const typename FEMTree< sizeof ... (FEMSigs), Real >::template DensityEstimator< WEIGHT_DEGREE >* density,
	const InputSampleDataType& zeroInputSampleDataType,
	SetVertexFunction SetVertex
)
{
	static const int Dim = sizeof ... (FEMSigs);
	typedef UIntPack< FEMSigs ... > Sigs;
	typedef typename VertexFactory::VertexType Vertex;

	static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE, BOUNDARY_FREE >::Signature;
	typedef typename FEMTree< Dim, Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;

	StreamingMesh< typename VertexFactory::VertexType, node_index_type >* mesh = new VectorStreamingMesh< Vertex, node_index_type >();

	typename LevelSetExtractor< Dim, Real, Vertex >::Stats stats;
	if (sampleData)
	{
		SparseNodeData< ProjectiveData< InputSampleDataType, Real >, IsotropicUIntPack< Dim, DataSig > > _sampleData = tree.template setExtrapolatedDataField< DataSig, false >(*samples, *sampleData, (DensityEstimator*)NULL);
		auto nodeFunctor = [&](const RegularTreeNode< Dim, FEMTreeNodeData, depth_and_offset_type >* n)
		{
			ProjectiveData< InputSampleDataType, Real >* clr = _sampleData(n);
			if (clr)
			{
				(*clr) *= static_cast<Real>(pow(params.colorPullFactor, tree.depth(n)));
			}
		};
		tree.tree().processNodes(nodeFunctor);
		stats = LevelSetExtractor< Dim, Real, Vertex >::template Extract< InputSampleDataType >(Sigs(), UIntPack< WEIGHT_DEGREE >(), UIntPack< DataSig >(), tree, density, &_sampleData, solution, isoValue, *mesh, zeroInputSampleDataType, SetVertex, !params.linearFit, false, !params.nonManifold, false, false);
	}
#if defined( __GNUC__ ) && __GNUC__ < 5
	else stats = LevelSetExtractor< Dim, Real, Vertex >::template Extract< InputSampleDataType >(Sigs(), UIntPack< WEIGHT_DEGREE >(), UIntPack< DataSig >(), tree, density, (SparseNodeData< ProjectiveData< InputSampleDataType, Real >, IsotropicUIntPack< Dim, DataSig > > *)NULL, solution, isoValue, *mesh, zeroInputSampleDataType, SetVertex, !params.linearFit, false, !params.nonManifold, false, false);
#else // !__GNUC__ || __GNUC__ >=5
	else stats = LevelSetExtractor< Dim, Real, Vertex >::template Extract< InputSampleDataType >(Sigs(), UIntPack< WEIGHT_DEGREE >(), UIntPack< DataSig >(), tree, density, NULL, solution, isoValue, *mesh, zeroInputSampleDataType, SetVertex, !params.linearFit, false, !params.nonManifold, false, false);
#endif // __GNUC__ || __GNUC__ < 4

	mesh->resetIterator();

	return mesh;
}

template< class Real, typename Data >
class CCPointStream : public InputDataStream<Data>
{
public:
	CCPointStream(const PoissonReconLib::ICloud<Real>& _cloud)
		: cloud(_cloud)
		, xform(nullptr)
		, currentIndex(0)
	{}

	void reset(void) override
	{
		currentIndex = 0;
	}

	bool next(Data& p) override
	{
		if (currentIndex >= cloud.size())
		{
			return false;
		}
		Point<Real, 3>& point = p.template get<0>();
		cloud.getPoint(currentIndex, point.coords);

		if (xform != nullptr)
		{
			point = (*xform) * point;
		}

		Point<Real, 3>& normal = p.template get<1>().template get<0>();
		cloud.getNormal(currentIndex, normal.coords);

		Point<float, 3>& color = p.template get<1>().template get<1>();
		if (cloud.hasColors())
		{
			cloud.getColor(currentIndex, color.coords);
		}
		else
		{
			color.coords[0] = color.coords[1] = color.coords[2] = 0;
		}

		currentIndex++;
		return true;
	}

public:
	const PoissonReconLib::ICloud<Real>& cloud;
	XForm<Real, 4>* xform;
	size_t currentIndex;
};

template< class Real, typename AuxDataFactory, unsigned int ... FEMSigs >
bool Execute(const PoissonReconLib::ICloud<Real>& in_cloud,
			 PoissonReconLib::IMesh<Real>& out_mesh,
			 const PoissonReconLib::Parameters& inputParams,
             UIntPack< FEMSigs ... >,
             const AuxDataFactory &auxDataFactory)
{
	// local copy of the parameters as they may be modified later
	PoissonReconLib::Parameters params = inputParams;

	static const int Dim = sizeof ... (FEMSigs);
	typedef UIntPack< FEMSigs ... > Sigs;
	typedef UIntPack< FEMSignature< FEMSigs >::Degree ... > Degrees;
	typedef UIntPack< FEMDegreeAndBType< NORMAL_DEGREE, DerivativeBoundary< FEMSignature< FEMSigs >::BType, 1 >::BType >::Signature ... > NormalSigs;
	static const unsigned int DataSig = FEMDegreeAndBType< DATA_DEGREE, BOUNDARY_FREE >::Signature;
	typedef typename FEMTree< Dim, Real >::template DensityEstimator< WEIGHT_DEGREE > DensityEstimator;
	typedef typename FEMTree< Dim, Real >::template InterpolationInfo< Real, 0 > InterpolationInfo;
	using namespace VertexFactory;

	// The factory for constructing an input sample
	typedef Factory< Real, PositionFactory< Real, Dim >, Factory< Real, NormalFactory< Real, Dim >, AuxDataFactory > > InputSampleFactory;

	// The factory for constructing an input sample's data
	typedef Factory< Real, NormalFactory< Real, Dim >, AuxDataFactory > InputSampleDataFactory;

	// The input point stream information: First piece of data is the normal; the remainder is the auxiliary data
	typedef InputOrientedPointStreamInfo< Real, Dim, typename AuxDataFactory::VertexType > InputPointStreamInfo;

	// The type of the input sample
	typedef typename InputPointStreamInfo::PointAndDataType InputSampleType;

	// The type of the input sample's data
	typedef typename InputPointStreamInfo::DataType InputSampleDataType;

	typedef            InputDataStream< InputSampleType >  InputPointStream;
	typedef TransformedInputDataStream< InputSampleType > XInputPointStream;

	InputSampleFactory inputSampleFactory(PositionFactory< Real, Dim >(), InputSampleDataFactory(NormalFactory< Real, Dim >(), auxDataFactory));
	InputSampleDataFactory inputSampleDataFactory(NormalFactory< Real, Dim >(), auxDataFactory);

	typedef RegularTreeNode< Dim, FEMTreeNodeData, depth_and_offset_type > FEMTreeNode;
	typedef typename FEMTreeInitializer< Dim, Real >::GeometryNodeType GeometryNodeType;

	bool needNormalData = (params.colorPullFactor > 0);
	bool needAuxData = (params.colorPullFactor > 0) && (auxDataFactory.bufferSize());

	XForm< Real, Dim + 1 > modelToUnitCube = XForm< Real, Dim + 1 >::Identity();

	Real isoValue = 0;

	FEMTree< Dim, Real > tree(MEMORY_ALLOCATOR_BLOCK_SIZE);

	if (params.depth > 0 && params.finestCellWidth > 0)
	{
		//Both the depth and the finest cell width are set, ignoring the latter
		params.finestCellWidth = 0.0f;
	}

	size_t pointCount = 0;

	std::vector< typename FEMTree< Dim, Real >::PointSample >* samples = new std::vector< typename FEMTree< Dim, Real >::PointSample >();
	std::vector< InputSampleDataType >* sampleData = new std::vector< InputSampleDataType >();
	Real targetValue = static_cast<Real>(0.5);

	InputSampleType P;

	// Read in the samples (and color data)
	{
		CCPointStream< Real, InputSampleType > pointStream(in_cloud);

		if (params.scale > 0.0f)
		{
			typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();

			typename InputSampleFactory::Transform _modelToUnitCube(modelToUnitCube);
			auto XFormFunctor = [&](InputSampleType &p) { _modelToUnitCube.inPlace(p); };
			XInputPointStream _pointStream(XFormFunctor, pointStream);
			modelToUnitCube = GetPointXForm< Real, Dim, typename AuxDataFactory::VertexType >(_pointStream, zeroData, static_cast<Real>(params.scale)) * modelToUnitCube;
		}

		if (params.finestCellWidth > 0)
		{
			double maxScale = 0;
			for (unsigned int i = 0; i < Dim; i++)
			{
				maxScale = std::max< double >(maxScale, 1.0 / modelToUnitCube(i, i));
			}
			params.depth = static_cast<unsigned int>(ceil(std::max< double >(0.0, log(maxScale / params.finestCellWidth) / log(2.0))));
		}
		//if (SolveDepth.value > params.depth)
		//{
		//	WARN("Solution depth cannot exceed system depth: ", SolveDepth.value, " <= ", Depth.value);
		//	SolveDepth.value = Depth.value;
		//}
		if (params.fullDepth > params.depth)
		{
			// Full depth cannot exceed system depth
			params.fullDepth = params.depth;
		}
		if (params.baseDepth > params.fullDepth)
		{
			//Base depth must be smaller than full depth
			params.baseDepth = params.fullDepth;
		}

		{
			typename InputSampleFactory::Transform _modelToUnitCube(modelToUnitCube);
			auto XFormFunctor = [&](InputSampleType &p) { _modelToUnitCube.inPlace(p); };
			XInputPointStream _pointStream(XFormFunctor, pointStream);
			auto ProcessData = [](const Point< Real, Dim > &p, typename InputPointStreamInfo::DataType &d)
			{
				Real l = static_cast<Real>(Length(d.template get<0>()));
				if (!l || !std::isfinite(l))
					return static_cast<Real>(-1.0);
				d.template get<0>() /= l;
				return static_cast<Real>(1.0);
			};

			typename InputSampleDataFactory::VertexType zeroData = inputSampleDataFactory();
			typename FEMTreeInitializer< Dim, Real >::StreamInitializationData sid;

			pointCount = FEMTreeInitializer< Dim, Real >::template Initialize< InputSampleDataType >(sid, tree.spaceRoot(), _pointStream, zeroData, params.depth, *samples, *sampleData, true, tree.nodeAllocators[0], tree.initializer(), ProcessData);
		}
	}

	XForm< Real, Dim + 1 > unitCubeToModel = modelToUnitCube.inverse();

	DenseNodeData< Real, Sigs > solution;
	ProjectiveData< Point< Real, 2 >, Real > pointDepthAndWeight;
	DenseNodeData< GeometryNodeType, IsotropicUIntPack< Dim, FEMTrivialSignature > > geometryNodeDesignators;
	DensityEstimator* density = nullptr;
	SparseNodeData< Point< Real, Dim >, NormalSigs >* normalInfo = nullptr;
	{
		DenseNodeData< Real, Sigs > constraints;
		InterpolationInfo* iInfo = nullptr;
		int solveDepth = static_cast<int>(params.depth);
		int splatDepth = static_cast<int>(params.depth) - 2;

		tree.resetNodeIndices(0, std::make_tuple());

		// Get the kernel density estimator
		{
			density = tree.template setDensityEstimator< 1, WEIGHT_DEGREE >(*samples, splatDepth, static_cast<Real>(params.samplesPerNode));
		}

		// Transform the Hermite samples into a vector field
		{
			normalInfo = new SparseNodeData< Point< Real, Dim >, NormalSigs >();
			std::function< bool(InputSampleDataType, Point< Real, Dim >&) > ConversionFunction = [](InputSampleDataType in, Point< Real, Dim > &out)
			{
				Point< Real, Dim > n = in.template get<0>();
				Real l = static_cast<Real>(Length(n));
				// It is possible that the samples have non-zero normals but there are two co-located samples with negative normals...
				if (!l)
					return false;
				out = n / l;
				return true;
			};

			*normalInfo = tree.setInterpolatedDataField(NormalSigs(), *samples, *sampleData, density, params.baseDepth, params.depth, static_cast<Real>(params.lowDepthCutOff), pointDepthAndWeight, ConversionFunction);

			ThreadPool::Parallel_for(0, normalInfo->size(), [&](unsigned int, size_t i) { (*normalInfo)[i] *= static_cast<Real>(-1.0); } );
		}

		if (!params.density)
		{
			delete density;
			density = nullptr;
		}

		if (!needNormalData && !needAuxData)
		{
			delete sampleData;
			sampleData = nullptr;
		}

		// Add the interpolation constraints
		if (params.pointWeight > 0)
		{
			if (params.exactInterpolation)
				iInfo = FEMTree< Dim, Real >::template       InitializeExactPointInterpolationInfo< Real, 0 >(tree, *samples, ConstraintDual< Dim, Real >(targetValue, static_cast<Real>(params.pointWeight) * pointDepthAndWeight.value()[1]), SystemDual< Dim, Real >(static_cast<Real>(params.pointWeight) * pointDepthAndWeight.value()[1]), true, false);
			else
				iInfo = FEMTree< Dim, Real >::template InitializeApproximatePointInterpolationInfo< Real, 0 >(tree, *samples, ConstraintDual< Dim, Real >(targetValue, static_cast<Real>(params.pointWeight) * pointDepthAndWeight.value()[1]), SystemDual< Dim, Real >(static_cast<Real>(params.pointWeight) * pointDepthAndWeight.value()[1]), true, params.depth, 1);
		}

		// Trim the tree and prepare for multigrid
		{
			constexpr int MAX_DEGREE = NORMAL_DEGREE > Degrees::Max() ? NORMAL_DEGREE : Degrees::Max();
			typename FEMTree< Dim, Real >::template HasNormalDataFunctor< NormalSigs > hasNormalDataFunctor(*normalInfo);
			auto hasDataFunctor = [&](const FEMTreeNode *node) { return hasNormalDataFunctor(node); };
			auto addNodeFunctor = [&](int d, const int off[Dim]) { return d <= params.fullDepth; };
			if (geometryNodeDesignators.size())
				tree.template finalizeForMultigridWithDirichlet< MAX_DEGREE, Degrees::Max() >(params.baseDepth, addNodeFunctor, hasDataFunctor, [&](const FEMTreeNode *node) { return node->nodeData.nodeIndex < (node_index_type)geometryNodeDesignators.size() && geometryNodeDesignators[node] == GeometryNodeType::EXTERIOR; }, std::make_tuple(iInfo), std::make_tuple(normalInfo, density, &geometryNodeDesignators));
			else
				tree.template finalizeForMultigrid< MAX_DEGREE, Degrees::Max() >(params.baseDepth, addNodeFunctor, hasDataFunctor, std::make_tuple(iInfo), std::make_tuple(normalInfo, density));
		}

		// Add the FEM constraints
		{
			constraints = tree.initDenseNodeData(Sigs());
			typename FEMIntegrator::template Constraint< Sigs, IsotropicUIntPack< Dim, 1 >, NormalSigs, IsotropicUIntPack< Dim, 0 >, Dim > F;
			unsigned int derivatives2[Dim];
			for (int d = 0; d < Dim; d++)
				derivatives2[d] = 0;
			typedef IsotropicUIntPack< Dim, 1 > Derivatives1;
			typedef IsotropicUIntPack< Dim, 0 > Derivatives2;
			for (int d = 0; d < Dim; d++)
			{
				unsigned int derivatives1[Dim];
				for (int dd = 0; dd < Dim; dd++)
					derivatives1[dd] = dd == d ? 1 : 0;
				F.weights[d][TensorDerivatives< Derivatives1 >::Index(derivatives1)][TensorDerivatives< Derivatives2 >::Index(derivatives2)] = 1;
			}
			tree.addFEMConstraints(F, *normalInfo, constraints, solveDepth);
		}

		// Free up the normal info
		delete normalInfo;
		normalInfo = nullptr;

		// Add the interpolation constraints
		if (params.pointWeight > 0)
		{
			tree.addInterpolationConstraints(constraints, solveDepth, std::make_tuple(iInfo));
		}

		// Solve the linear system
		{
			typename FEMTree< Dim, Real >::SolverInfo sInfo;
			sInfo.cgDepth = 0;
			sInfo.cascadic = true;
			sInfo.vCycles = 1;
			sInfo.iters = params.iters;
			sInfo.cgAccuracy = params.cgAccuracy;
			sInfo.verbose = false;
			sInfo.showResidual = false;
			sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE;
			sInfo.sliceBlockSize = 1;
			sInfo.baseVCycles = params.baseVCycles;
			typename FEMIntegrator::template System< Sigs, IsotropicUIntPack< Dim, 1 > > F({ 0. , 1. });
			solution = tree.solveSystem(Sigs(), F, constraints, params.baseDepth, params.depth, sInfo, std::make_tuple(iInfo));
			if (iInfo)
			{
				delete iInfo;
				iInfo = nullptr;
			}
		}
	}

	{
		double valueSum = 0, weightSum = 0;
		typename FEMTree< Dim, Real >::template MultiThreadedEvaluator< Sigs, 0 > evaluator(&tree, solution);
		std::vector< double > valueSums(ThreadPool::NumThreads(), 0), weightSums(ThreadPool::NumThreads(), 0);
		ThreadPool::Parallel_for(0, samples->size(), [&](unsigned int thread, size_t j)
			{
				ProjectiveData< Point< Real, Dim >, Real >& sample = (*samples)[j].sample;
				Real w = sample.weight;
				if (w > 0)
					weightSums[thread] += w, valueSums[thread] += evaluator.values(sample.data / sample.weight, thread, (*samples)[j].node)[0] * w;
			}
		);
		for (size_t t = 0; t < valueSums.size(); t++)
			valueSum += valueSums[t], weightSum += weightSums[t];
		isoValue = static_cast<Real>(valueSum / weightSum);

		if (!needNormalData && !needAuxData)
		{
			delete samples;
			samples = nullptr;
		}
	}

	// Extract the output mesh
	if (params.density)
	{
		typedef Factory< Real, PositionFactory< Real, Dim >, NormalFactory< Real, Dim >, ValueFactory< Real >, AuxDataFactory > VertexFactory;
		VertexFactory vertexFactory(PositionFactory< Real, Dim >(), NormalFactory< Real, Dim >(), ValueFactory< Real >(), auxDataFactory);

		auto SetVertex = [](typename VertexFactory::VertexType& v, const Point< Real, Dim >& p, Point< Real, Dim >& g, Real w, const InputSampleDataType& d)
		{
			v.template get<0>() = p;
			v.template get<1>() = d.template get<0>();
			v.template get<2>() = w;
			v.template get<3>() = d.template get<1>();
		};

		auto mesh = ExtractMesh<VertexFactory>(params, UIntPack< FEMSigs ... >(), tree, solution, isoValue, samples, sampleData, density, inputSampleDataFactory(), SetVertex);
		if (mesh)
		{
			typename VertexFactory::Transform unitCubeToModelTransform(unitCubeToModel);
			auto xForm = [&](typename VertexFactory::VertexType& v) { unitCubeToModelTransform.inPlace(v); };

			// write vertices
			{
				typename VertexFactory::VertexType vertex = vertexFactory();
				for (size_t i = 0; i < mesh->vertexNum(); i++)
				{
					mesh->nextVertex(vertex);
					xForm(vertex);
					out_mesh.addVertex(vertex.template get<0>().coords);

					const auto* normal = vertex.template get<1>().coords;
					out_mesh.addNormal(normal);

					auto density = vertex.template get<2>();
					out_mesh.addDensity(density);

					const auto* color = vertex.template get<3>().coords;
					out_mesh.addColor(color);
				}
			}

			// write faces
			{
				std::vector<int> polygon;
				for (size_t i = 0; i < mesh->polygonNum(); i++)
				{
					mesh->nextPolygon(polygon);
					if (polygon.size() != 3)
					{
						//unsupported
						assert(false);
						delete mesh;
						return false;
					}
					out_mesh.addTriangle(polygon[0], polygon[1], polygon[2]);
				}
			}

			delete mesh;
			mesh = nullptr;
		}
	}
	else
	{
		typedef Factory< Real, PositionFactory< Real, Dim >, NormalFactory< Real, Dim >, AuxDataFactory > VertexFactory;
		VertexFactory vertexFactory(PositionFactory< Real, Dim >(), NormalFactory< Real, Dim >(), auxDataFactory);

		auto SetVertex = [](typename VertexFactory::VertexType& v, const Point< Real, Dim >& p, const Point< Real, Dim >& g, Real w, const InputSampleDataType& d)
		{
			v.template get<0>() = p;
			v.template get<1>() = d.template get<0>();
			v.template get<2>() = d.template get<1>();
		};

		auto mesh = ExtractMesh<VertexFactory>(params, UIntPack< FEMSigs ... >(), tree, solution, isoValue, samples, sampleData, density, inputSampleDataFactory(), SetVertex);
		if (mesh)
		{
			typename VertexFactory::Transform unitCubeToModelTransform(unitCubeToModel);
			auto xForm = [&](typename VertexFactory::VertexType& v) { unitCubeToModelTransform.inPlace(v); };

			// write vertices
			{
				typename VertexFactory::VertexType vertex = vertexFactory();
				for (size_t i = 0; i < mesh->vertexNum(); i++)
				{
					mesh->nextVertex(vertex);
					xForm(vertex);
					out_mesh.addVertex(vertex.template get<0>().coords);

					const auto* normal = vertex.template get<1>().coords;
					out_mesh.addNormal(normal);

					const auto* color = vertex.template get<2>().coords;
					out_mesh.addColor(color);
				}
			}

			// write faces
			{
				std::vector<int> polygon;
				for (size_t i = 0; i < mesh->polygonNum(); i++)
				{
					mesh->nextPolygon(polygon);
					if (polygon.size() != 3)
					{
						//unsupported
						assert(false);
						delete mesh;
						return false;
					}
					out_mesh.addTriangle(polygon[0], polygon[1], polygon[2]);
				}
			}

			delete mesh;
			mesh = nullptr;
		}
	}

	if (density)
	{
		delete density;
		density = nullptr;
	}

	return true;
}

bool PoissonReconLib::Reconstruct(	const Parameters& params,
									const ICloud<float>& inCloud,
									IMesh<float>& outMesh )
{
	if (!inCloud.hasNormals())
	{
		//we need normals
		return false;
	}

#ifdef WITH_OPENMP
	ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP, params.threads);
#else
	ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL, params.threads);
#endif

	bool success = false;

	typedef VertexFactory::RGBColorFactory<float> AuxDataFactory;

#if 0
	typedef VertexFactory::EmptyFactory<float> EmptyAuxDataFactory;
	typedef VertexFactory::RGBColorFactory<uint8_t> RGBAuxDataFactory;
	typedef VertexFactory::ValueFactory<float> FloatScalarFactory;

	// The input point stream information: First piece of data is the normal; the remainder is the auxiliary data
	typedef InputOrientedPointStreamInfo< float, DIMENSION, typename AuxDataFactory::VertexType > InputPointStreamInfoTest;
	typedef VertexFactory::NormalFactory< float, DIMENSION >::VertexType NormalType;
	typedef VectorTypeUnion< float, typename NormalType, RGBAuxDataFactory, EmptyAuxDataFactory > MetaVertex;

	MetaVertex test;
	test.template get<0>().coords; // normal
	test.template get<1>()().coords; // color
	test.template get<2>()(); // scalar

	InputOrientedPointStreamInfo<float, DIMENSION, AuxDataFactory>::PointAndDataType test2;
	test2.template get<0>().coords;
	test2.template get<1>().template get<0>().coords;
	test2.template get<1>().template get<1>()();

	//Point<float, DIMENSION>::coords

#endif

	switch (params.boundary)
	{
	case Parameters::FREE:
		success = Execute<float, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_FREE>::Signature>(), AuxDataFactory());
		break;
	case Parameters::DIRICHLET:
		success = Execute<float, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_DIRICHLET>::Signature>(), AuxDataFactory());
		break;
	case Parameters::NEUMANN:
		success = Execute<float, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_NEUMANN>::Signature>(), AuxDataFactory());
		break;
	default:
		assert(false);
		break;
	}

	ThreadPool::Terminate();

	return success;
}

bool PoissonReconLib::Reconstruct(	const Parameters& params,
									const ICloud<double>& inCloud,
									IMesh<double>& outMesh )
{
	if (!inCloud.hasNormals())
	{
		//we need normals
		return false;
	}

#ifdef WITH_OPENMP
	ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::OPEN_MP, params.threads);
#else
	ThreadPool::Init((ThreadPool::ParallelType)(int)ThreadPool::THREAD_POOL, params.threads);
#endif

	bool success = false;

	typedef VertexFactory::RGBColorFactory<float> AuxDataFactory;

	switch (params.boundary)
	{
	case Parameters::FREE:
		success = Execute<double, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_FREE>::Signature>(), AuxDataFactory());
		break;
	case Parameters::DIRICHLET:
		success = Execute<double, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_DIRICHLET>::Signature>(), AuxDataFactory());
		break;
	case Parameters::NEUMANN:
		success = Execute<double, AuxDataFactory>(inCloud, outMesh, params, IsotropicUIntPack<DIMENSION, FEMDegreeAndBType<DEFAULT_FEM_DEGREE, BOUNDARY_NEUMANN>::Signature>(), AuxDataFactory());
		break;
	default:
		assert(false);
		break;
	}

	ThreadPool::Terminate();

	return success;

}
