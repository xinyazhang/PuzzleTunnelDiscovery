#ifndef PYSE3OMPL_OMPLDRIVER_H
#define PYSE3OMPL_OMPLDRIVER_H

#include <limits>
#include <memory>
#include <stdint.h>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <omplapp/apps/SE3RigidBodyPlanning.h>
#include <ompl/geometric/planners/rrt/ReRRT.h>
#include <ompl/tools/config/SelfConfig.h>
//#include <omplapp/geometry/detail/FCLStateValidityChecker.h>
#include <omplapp/geometry/detail/FCLContinuousMotionValidator.h>
#include <omplapp/config.h>
#include "config_planner.h"
#include <iostream>
#include <vector>
#include <unordered_set>

enum {
	MODEL_PART_ENV = 0,
	MODEL_PART_ROB = 1,
	TOTAL_MODEL_PARTS
};

enum {
	INIT_STATE = 0,
	GOAL_STATE = 1,
	TOTAL_STATES
};


class OmplDriver {
	struct SE3State {
		Eigen::Vector3d tr;
		Eigen::Vector3d rot_axis;
		double rot_angle;
	};
public:
	using GraphV = Eigen::MatrixXd;
	using GraphVFlags = Eigen::Matrix<uint32_t, -1, 1>;
	using GraphE = Eigen::SparseMatrix<uint8_t>;
	using Motion = ompl::geometric::ReRRT::Motion;
	using NearestNeighbors = ompl::NearestNeighbors<Motion*>;
	using KNNPtr = std::shared_ptr<NearestNeighbors>;

	struct PerformanceNumbers {
		double planning_time = 0;
		unsigned long motion_check = 0;
		double motion_check_time = 0;
		unsigned long motion_discrete_state_check = 0;
		double knn_query_time = 0;
		double knn_delete_time = 0;
	};

	OmplDriver()
	{
	}

	~OmplDriver()
	{
	}

	void setPlanner(int planner_id,
			int valid_state_sampler_id, // WARNING: SOME PLANNERS ARE NOT BASED ON VALID STATE SAMPLER
			std::string sample_injection_fn,
			int rdt_k_nearest)
	{
		planner_id_ = planner_id;
		vs_sampler_id_ = valid_state_sampler_id;
		sample_inj_fn_ = sample_injection_fn;
		rdt_k_nearest_ = rdt_k_nearest;
	}

	void setModelFile(int part_id, const std::string& fn)
	{
		if (part_id < 0 || part_id >= TOTAL_MODEL_PARTS) {
			throw std::runtime_error("OmplDriver::setModelFile: invalid part_id "
			                         + std::to_string(part_id));
		}
		model_files_[part_id] = fn;
	}

	// set the state from R^3 translation and axis angle rotations.
	void setState(int state_type,
	              const Eigen::Vector3d& tr,
	              const Eigen::Vector3d& rot_axis,
	              double rot_angle)
	{
		if (state_type < 0 || state_type >= TOTAL_STATES) {
			throw std::runtime_error("OmplDriver::setState: invalid state_type "
			                         + std::to_string(state_type));
		}
		auto& m = real_problem_states_[state_type];
		m.tr = tr;
		m.rot_axis = rot_axis;
		m.rot_angle = rot_angle;
		problem_states_[state_type] = m;
	}

	// set the Bounding Box
	void setBB(const Eigen::Vector3d& mins,
	           const Eigen::Vector3d& maxs)
	{
		mins_ = mins;
		maxs_ = maxs;
	}

	// Collision detection resolution
	void setCDRes(double cdres) { cdres_ = cdres; }

	// Set the option vector.
	// Option vector is a list of strings designed to pass arguments to
	// motion planners in an end-to-end manner
	void setOptionVector(std::vector<std::string> ovec) { option_vector_ = std::move(ovec); }

	// Solve the puzzle
	std::tuple<GraphV, GraphE>
	solve(double days,
	      const std::string& output_fn,
	      bool return_ve = false,
	      int_least64_t sbudget = -1,
	      bool record_compact_tree = false,
	      bool continuous = false);

	// Sample nsamples uniformly within C-space
	GraphV presample(size_t nsamples);

	// NOTE: TRANSLATION + W-LAST QUATERNION
	void substituteState(int state_type, const Eigen::VectorXd& state)
	{
		if (state_type < 0 || state_type >= TOTAL_STATES) {
			throw std::runtime_error("OmplDriver::substituteState: invalid state_type "
			                         + std::to_string(state_type));
		}
		if (state.size() == 0) {
			// reset the state
			problem_states_[state_type] = real_problem_states_[state_type];
			return ;
		}
		Eigen::Vector3d tr;
		tr << state(0), state(1), state(2);
		Eigen::Quaternion<double> quat(state(6), state(3), state(4), state(5));
		Eigen::AngleAxis<double> aa(quat);

		auto& m = problem_states_[state_type];
		m.tr = tr;
		m.rot_axis = aa.axis();
		m.rot_angle = aa.angle();
	}

	void addExistingGraph(GraphV V, GraphE E)
	{
		ex_graph_v_.emplace_back(std::move(V));
		ex_graph_e_.emplace_back(std::move(E));
	}

	//
	// Merge graphs added by addExistingGraph
	//
	// Param
	//   KNN: K-Nearest Neighbors
	//
	// Returns
	//   a Nx4 integer matrix, each row is composed by
	//   (from graph id, from vertex id in graph, to graph id, to vertex id in graph)
	//   all IDs are 0-indexed
	//
	// Note: ex_graph_e_ will not be used, assuming each graph is connected.
	Eigen::MatrixXi
	mergeExistingGraph(int KNN,
	                   bool verbose = false,
	                   int version = 0,
			   Eigen::VectorXi subset = Eigen::VectorXi());

	Eigen::VectorXi
	validateStates(const Eigen::MatrixXd& qs0);

	Eigen::VectorXi
	validateMotionPairs(const Eigen::MatrixXd& qs0,
			    const Eigen::MatrixXd& qs1);

	// Only one set is supported. If multiple ones present, users are
	// supposed to call to merge them together in python side, which is
	// eaiser
	void setSampleSet(GraphV Q)
	{
		predefined_sample_set_ = std::move(Q);
	}

	void setSampleSetEdges(Eigen::MatrixXi QB, Eigen::MatrixXi QE, Eigen::MatrixXi QEB)
	{
		pds_tree_bases_ = std::move(QB);
		pds_edges_ = std::move(QE);
		pds_edge_bases_ = std::move(QEB);
	}

	void setSampleSetFlags(GraphVFlags QF)
	{
		pds_flags_ = std::move(QF);
	}

	Eigen::SparseMatrix<int>
	getSampleSetConnectivity() const
	{
		return predefined_set_connectivity_;
	}

	std::tuple<
		Eigen::Matrix<int64_t, -1, 1>,
		Eigen::MatrixXd,
		Eigen::Matrix<int64_t, -1, 2>
	          >
	getCompactGraph() const
	{
		return std::tie(compact_nouveau_vertex_id_,
				compact_nouveau_vertices_,
				compact_edges_);
	}

	Eigen::MatrixXd
	getLatestSolution() const
	{
		return latest_solution_;
	}

	int
	getLatestSolutionStatus() const
	{
		return static_cast<int>(latest_solution_status_);
	}

	Eigen::VectorXi
	getGraphIStateIndices() const
	{
		return graph_istate_indices_;
	}

	Eigen::VectorXi
	getGraphGStateIndices() const
	{
		return graph_gstate_indices_;
	}

	PerformanceNumbers
	getLatestPerformanceNumbers() const
	{
		return latest_pn_;
	}
private:
	int planner_id_;
	int vs_sampler_id_;
	int rdt_k_nearest_;
	std::string sample_inj_fn_;
	std::string model_files_[TOTAL_MODEL_PARTS];
	SE3State real_problem_states_[TOTAL_STATES];
	SE3State problem_states_[TOTAL_STATES]; // Effective states
	Eigen::Vector3d mins_, maxs_;
	double cdres_ = std::numeric_limits<double>::quiet_NaN();

	std::vector<GraphV> ex_graph_v_;
	std::vector<GraphE> ex_graph_e_;

	GraphV predefined_sample_set_;
	Eigen::MatrixXi pds_tree_bases_;
	Eigen::MatrixXi pds_edges_;
	Eigen::MatrixXi pds_edge_bases_;
	GraphVFlags pds_flags_;
	Eigen::SparseMatrix<int> predefined_set_connectivity_;
	std::vector<std::string> option_vector_;

	void configSE3RigidBodyPlanning(ompl::app::SE3RigidBodyPlanning& setup, bool continuous = false);

	Eigen::Matrix<int64_t, -1, 1> compact_nouveau_vertex_id_;
	Eigen::MatrixXd compact_nouveau_vertices_;
	Eigen::Matrix<int64_t, -1, 2> compact_edges_;

	void recordCompactTree(ompl::base::PlannerPtr planner)
	{
		planner->getCompactGraph(compact_nouveau_vertex_id_,
		                         compact_nouveau_vertices_,
		                         compact_edges_);
	}

	Eigen::MatrixXd latest_solution_;
	ompl::base::PlannerStatus::StatusType latest_solution_status_;

	Eigen::VectorXi graph_istate_indices_;
	Eigen::VectorXi graph_gstate_indices_;

	// Create a KNN in the same manner as of RDT (ompl::geometric::ReRRT)
	// Note: different planner may have different metrics
	std::shared_ptr<NearestNeighbors>
	createKNNForRDT(const ompl::geometric::ReRRT* planner)
	{
		using ompl::geometric::ReRRT;
		std::shared_ptr<NearestNeighbors> ret;
		ret.reset(ompl::tools::SelfConfig::getDefaultNearestNeighbors<Motion*>(planner));
		ret->setDistanceFunction(std::bind(&ReRRT::distanceFunction,
					           planner,
						   std::placeholders::_1,
						   std::placeholders::_2));
		return ret;
	}

	std::vector<KNNPtr> ex_knn_;

	PerformanceNumbers latest_pn_;

	void updatePerformanceNumbers(ompl::app::SE3RigidBodyPlanning& setup);
};

#endif
