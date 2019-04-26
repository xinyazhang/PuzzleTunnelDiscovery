#include <limits>
#include <stdint.h>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <string>
#include <omplapp/apps/SE3RigidBodyPlanning.h>
//#include <omplapp/geometry/detail/FCLStateValidityChecker.h>
#include <omplapp/geometry/detail/FCLContinuousMotionValidator.h>
#include <omplapp/config.h>
#include "config_planner.h"
#include <iostream>
#include <vector>
namespace py = pybind11;

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

	void setState(int state_type,
	              const Eigen::Vector3d& tr,
	              const Eigen::Vector3d& rot_axis,
	              double rot_angle)
	{
		if (state_type < 0 || state_type >= TOTAL_STATES) {
			throw std::runtime_error("OmplDriver::setState: invalid state_type "
			                         + std::to_string(state_type));
		}
		auto& m = problem_states_[state_type];
		m.tr = tr;
		m.rot_axis = rot_axis;
		m.rot_angle = rot_angle;
	}

	void setBB(const Eigen::Vector3d& mins, 
	           const Eigen::Vector3d& maxs)
	{
		mins_ = mins;
		maxs_ = maxs;
	}

	// Collision detection resolution
	void setCDRes(double cdres) { cdres_ = cdres; }

	void setOptionVector(std::vector<std::string> ovec) { option_vector_ = std::move(ovec); }

	std::tuple<GraphV, GraphE>
	solve(double days,
	      const std::string& output_fn,
	      bool return_ve = false,
	      int_least64_t sbudget = -1,
	      bool record_compact_tree = false,
	      bool continuous = false)
	{
		using namespace ompl;

		ompl::app::SE3RigidBodyPlanning setup;
		configSE3RigidBodyPlanning(setup, continuous);
		std::cout << "Trying to solve "
		          << model_files_[MODEL_PART_ROB]
			  << " v.s. "
			  << model_files_[MODEL_PART_ENV]
			  << std::endl;
		setup.print();
		if (!ex_graph_v_.empty()) {
			auto planner = setup.getPlanner();
			for (size_t i = 0; i < ex_graph_v_.size(); i++) {
				planner->addGraph(ex_graph_v_[i], ex_graph_e_[i]);
			}
		}
		if (predefined_sample_set_.rows() > 0) {
			auto planner = setup.getPlanner();
			planner->setSampleSet(predefined_sample_set_);
			if (pds_flags_.rows() > 0) {
				planner->setSampleSetFlags(pds_flags_);
			}
		} else {
			if (record_compact_tree) {
				throw std::runtime_error("record_compact_tree requires set_sample_set");
			}
		}
		if (sbudget > 0)
			throw std::runtime_error("sbudget is not implemented");
		auto status = setup.solve(3600 * 24 * days);
		if (status) {
			std::cout.precision(17);
			setup.getSolutionPath().printAsMatrix(std::cout);
			setup.getSolutionPath().toMatrix(latest_solution_);
			latest_solution_status_ = status;
		}
		GraphV V;
		GraphE E;
		if (!output_fn.empty() or return_ve) {
			base::PlannerData pdata(setup.getSpaceInformation());
			setup.getPlanner()->getPlannerData(pdata);
			if (!output_fn.empty()) {
				std::ofstream fout(output_fn);
				fout.precision(17);
				printPlan(pdata, fout);
			}
			if (return_ve) {
				extractPlanVE(pdata, V, E);
			}
		}
		std::cout << "-----FINAL-----" << std::endl;
		ompl::base::Planner::PlannerProgressProperties props = setup.getPlanner()->getPlannerProgressProperties();
		std::cerr << "Final properties\n";
		for (const auto& item : props) {
			std::cerr << item.first << ": " << item.second() << std::endl;
		}
		if (predefined_sample_set_.rows() > 0) {
			setup.getPlanner()->getSampleSetConnectivity(predefined_set_connectivity_);
			if (record_compact_tree) {
				recordCompactTree(setup.getPlanner());
			}
		}
		return std::tie(V, E);
	}

	GraphV
	presample(size_t nsamples)
	{
		GraphV ret;
		ompl::app::SE3RigidBodyPlanning setup;
		configSE3RigidBodyPlanning(setup);
		auto ss = setup.getGeometricComponentStateSpace();
		auto sampler = ss->allocStateSampler();
		auto state = ss->allocState();

		std::vector<double> reals;
		ss->copyToReals(reals, state);
		ret.resize(nsamples, reals.size());
		for (size_t i = 0; i < nsamples; i++) {
			sampler->sampleUniform(state);
			ss->copyToReals(reals, state);
			ret.row(i) = Eigen::Map<Eigen::VectorXd>(reals.data(), reals.size());
		}

		ss->freeState(state);
		return ret;
	}

	// NOTE: TRANSLATION + W-LAST QUATERNION
	void substituteState(int state_type, const Eigen::VectorXd& state)
	{
		Eigen::Vector3d tr;
		tr << state(0), state(1), state(2);
		Eigen::Quaternion<double> quat(state(6), state(3), state(4), state(5));
		Eigen::AngleAxis<double> aa(quat);
		setState(state_type, tr, aa.axis(), aa.angle());
	}

	void addExistingGraph(GraphV V, GraphE E)
	{
		ex_graph_v_.emplace_back(std::move(V));
		ex_graph_e_.emplace_back(std::move(E));
	}

	// Only one set is supported. If multiple ones present, users are
	// supposed to call to merge them together in python side, which is
	// eaiser
	void setSampleSet(GraphV Q)
	{
		predefined_sample_set_ = std::move(Q);
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
private:
	int planner_id_;
	int vs_sampler_id_;
	int rdt_k_nearest_;
	std::string sample_inj_fn_;
	std::string model_files_[TOTAL_MODEL_PARTS];
	SE3State problem_states_[TOTAL_STATES];
	Eigen::Vector3d mins_, maxs_;
	double cdres_ = std::numeric_limits<double>::quiet_NaN();

	std::vector<GraphV> ex_graph_v_;
	std::vector<GraphE> ex_graph_e_;

	GraphV predefined_sample_set_;
	GraphVFlags pds_flags_;
	Eigen::SparseMatrix<int> predefined_set_connectivity_;
	std::vector<std::string> option_vector_;

	void configSE3RigidBodyPlanning(ompl::app::SE3RigidBodyPlanning& setup, bool continuous = false)
	{
		using namespace ompl;

		config_planner(setup,
			       planner_id_,
			       vs_sampler_id_,
			       sample_inj_fn_.c_str(),
			       rdt_k_nearest_);
		bool loaded;
		loaded = setup.setRobotMesh(model_files_[MODEL_PART_ROB]);
		loaded = loaded && setup.setEnvironmentMesh(model_files_[MODEL_PART_ENV]);
		if (!loaded) {
			throw std::runtime_error("Failed to load rob/env gemoetry");
		}

		auto& ist = problem_states_[INIT_STATE];
		base::ScopedState<base::SE3StateSpace> start(setup.getSpaceInformation());
		start->setX(ist.tr(0));
		start->setY(ist.tr(1));
		start->setZ(ist.tr(2));
		start->rotation().setAxisAngle(ist.rot_axis(0),
					       ist.rot_axis(1),
					       ist.rot_axis(2),
					       ist.rot_angle);
		auto& gst = problem_states_[GOAL_STATE];
		base::ScopedState<base::SE3StateSpace> goal(start);
		goal->setX(gst.tr(0));
		goal->setY(gst.tr(1));
		goal->setZ(gst.tr(2));
		goal->rotation().setAxisAngle(gst.rot_axis(0),
					      gst.rot_axis(1),
					      gst.rot_axis(2),
					      gst.rot_angle);
		setup.setStartAndGoalStates(start, goal);
		if (!continuous)
			setup.getSpaceInformation()->setStateValidityCheckingResolution(cdres_);

		auto gcss = setup.getGeometricComponentStateSpace()->as<base::SE3StateSpace>();
		base::RealVectorBounds b = gcss->getBounds();
		for (int i = 0; i < 3; i++) {
			b.setLow(i, mins_(i));
			b.setHigh(i, maxs_(i));
		}
		gcss->setBounds(b);
		if (!option_vector_.empty())
			setup.getPlanner()->setOptionVector(option_vector_);
		setup.setup();
		if (continuous) {
			// Note: we need to do this AFTER calling setup()
			//       since Motion Validator requires State
			//       Validator in the SpaceInformation object,
			//       whcih is done in setup()
			auto si = setup.getSpaceInformation();
			si->setMotionValidator(std::make_shared<app::FCLContinuousMotionValidator>(si.get(), app::Motion_3D));
			setup.setup();
		}
	}

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
};

PYBIND11_MODULE(pyse3ompl, m) {
	m.attr("PLANNER_RRT_CONNECT") = py::int_(int(PLANNER_RRT_CONNECT));
	m.attr("PLANNER_RRT"        ) = py::int_(int(PLANNER_RRT        ));
	m.attr("PLANNER_BKPIECE1"   ) = py::int_(int(PLANNER_BKPIECE1   ));
	m.attr("PLANNER_LBKPIECE1"  ) = py::int_(int(PLANNER_LBKPIECE1  ));
	m.attr("PLANNER_KPIECE1"    ) = py::int_(int(PLANNER_KPIECE1    ));
	m.attr("PLANNER_SBL"        ) = py::int_(int(PLANNER_SBL        ));
	m.attr("PLANNER_EST"        ) = py::int_(int(PLANNER_EST        ));
	m.attr("PLANNER_PRM"        ) = py::int_(int(PLANNER_PRM        ));
	m.attr("PLANNER_BITstar"    ) = py::int_(int(PLANNER_BITstar    ));
	m.attr("PLANNER_PDST"       ) = py::int_(int(PLANNER_PDST       ));
	m.attr("PLANNER_TRRT"       ) = py::int_(int(PLANNER_TRRT       ));
	m.attr("PLANNER_BiTRRT"     ) = py::int_(int(PLANNER_BiTRRT     ));
	m.attr("PLANNER_LazyRRT"    ) = py::int_(int(PLANNER_LazyRRT    ));
	m.attr("PLANNER_LazyLBTRRT" ) = py::int_(int(PLANNER_LazyLBTRRT ));
	m.attr("PLANNER_SPARS"      ) = py::int_(int(PLANNER_SPARS      ));
	m.attr("PLANNER_ReRRT"      ) = py::int_(int(PLANNER_ReRRT      ));
	m.attr("VSTATE_SAMPLER_UNIFORM") = py::int_(0);
	m.attr("VSTATE_SAMPLER_GAUSSIAN") = py::int_(1);
	m.attr("VSTATE_SAMPLER_OBSTACLE") = py::int_(2);
	m.attr("VSTATE_SAMPLER_MAXCLEARANCE") = py::int_(3);
	m.attr("VSTATE_SAMPLER_PROXY") = py::int_(4);
	m.attr("MODEL_PART_ENV") = py::int_(int(MODEL_PART_ENV));
	m.attr("MODEL_PART_ROB") = py::int_(int(MODEL_PART_ROB));
	m.attr("INIT_STATE") = py::int_(int(INIT_STATE));
	m.attr("GOAL_STATE") = py::int_(int(GOAL_STATE));
	m.attr("EXACT_SOLUTION") = py::int_(int(ompl::base::PlannerStatus::EXACT_SOLUTION));
	py::class_<OmplDriver>(m, "OmplDriver")
		.def(py::init<>())
		.def("set_planner", &OmplDriver::setPlanner)
		.def("set_model_file", &OmplDriver::setModelFile)
		.def("set_bb", &OmplDriver::setBB)
		.def("set_state", &OmplDriver::setState)
		.def("set_cdres", &OmplDriver::setCDRes)
		.def("set_option_vector", &OmplDriver::setOptionVector)
		.def("solve", &OmplDriver::solve,
		     py::arg("days"),
		     py::arg("output_fn") = std::string(),
		     py::arg("return_ve") = false,
		     py::arg("sbudget") = -1,
		     py::arg("record_compact_tree") = false,
		     py::arg("continuous_motion_validator") = false
		    )
		.def("substitute_state", &OmplDriver::substituteState)
		.def("add_existing_graph", &OmplDriver::addExistingGraph)
		.def("set_sample_set", &OmplDriver::setSampleSet)
		.def("set_sample_set_flags", &OmplDriver::setSampleSetFlags)
		.def("get_sample_set_connectivity", &OmplDriver::getSampleSetConnectivity)
		.def("presample", &OmplDriver::presample)
		.def("get_compact_graph", &OmplDriver::getCompactGraph)
		.def_property_readonly("latest_solution", &OmplDriver::getLatestSolution)
		.def_property_readonly("latest_solution_status", &OmplDriver::getLatestSolutionStatus)
		;
}
