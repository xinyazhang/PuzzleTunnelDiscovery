#include "ompldriver.h"
#include <chrono>

using hclock = std::chrono::high_resolution_clock;
using GraphV = OmplDriver::GraphV;
using GraphE = OmplDriver::GraphE;

std::tuple<GraphV, GraphE>
OmplDriver::solve(double days,
                  const std::string& output_fn,
		  bool return_ve,
		  int_least64_t sbudget,
		  bool record_compact_tree,
		  bool continuous)
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
		if (pds_tree_bases_.rows() > 0) {
			planner->setSampleSetEdges(pds_tree_bases_,
					pds_edges_,
					pds_edge_bases_);
		}
	} else {
		if (record_compact_tree) {
			throw std::runtime_error("record_compact_tree requires set_sample_set");
		}
	}
	if (sbudget > 0)
		throw std::runtime_error("sbudget is not implemented");
	latest_solution_.resize(0, 0);
	latest_solution_status_ = ompl::base::PlannerStatus::UNKNOWN;
	auto plan_start = hclock::now();
	auto status = setup.solve(3600 * 24 * days);
	std::chrono::duration<uint64_t, std::nano> plan_dur = hclock::now() - plan_start;
	latest_pn_.planning_time = plan_dur.count() * 1e-6;
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
			graph_istate_indices_.resize(pdata.numStartVertices());
			graph_gstate_indices_.resize(pdata.numGoalVertices());
			for (int i = 0; i < graph_istate_indices_.rows(); i++) {
				graph_istate_indices_(i) = pdata.getStartIndex(i);
			}
			for (int i = 0; i < graph_gstate_indices_.rows(); i++) {
				graph_gstate_indices_(i) = pdata.getGoalIndex(i);
			}
		}
	}
	std::cout << "-----FINAL-----" << std::endl;
	ompl::base::Planner::PlannerProgressProperties props = setup.getPlanner()->getPlannerProgressProperties();
	std::cerr << "Final properties\n";
	for (const auto& item : props) {
		std::cerr << item.first << ": " << item.second() << std::endl;
	}
	updatePerformanceNumbers(setup);
	if (predefined_sample_set_.rows() > 0) {
		setup.getPlanner()->getSampleSetConnectivity(predefined_set_connectivity_);
		if (record_compact_tree) {
			recordCompactTree(setup.getPlanner());
		}
	}
	return std::tie(V, E);
}


GraphV
OmplDriver::presample(size_t nsamples)
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


Eigen::MatrixXi
OmplDriver::mergeExistingGraph(int KNN,
                               bool verbose,
                               int version,
                               Eigen::VectorXi subset)
{
	// We do not really need an SE3RigidBodyPlanning object,
	// but this make things much easier
	ompl::app::SE3RigidBodyPlanning setup;
	{
		auto bak = planner_id_;
		planner_id_ = PLANNER_ReRRT;
		configSE3RigidBodyPlanning(setup, false);
		planner_id_ = bak;
	}
	auto generic_planner = setup.getPlanner();
	auto real_planner = std::dynamic_pointer_cast<ompl::geometric::ReRRT>(generic_planner);
	if (!real_planner) {
		// This should not happen.
		throw std::runtime_error("FATAL: SE3RigidBodyPlanning does not contain a ReRRT planner\n");
	}
	auto nn = real_planner->_accessNearestNeighbors();
	auto si = real_planner->getSpaceInformation();
	auto ss = si->getStateSpace();
	int last_pc = 0;
	std::vector<Motion*> all_motions; // DS to track all motions
	std::vector<Eigen::Vector4i> edges;
	{
		size_t ttl = 0;
		for (const auto& V: ex_graph_v_)
			ttl += V.rows();
		all_motions.reserve(ttl);
	}
	if (version == 0) {
		/*
		 * Put all blooming trees into single KNN DS
		 */
		for (size_t i = 0; i < ex_graph_v_.size(); i++) {/*{{{*/
			const auto& V = ex_graph_v_[i];
			for (int j = 0; j < V.rows(); j++) {
				auto m = new Motion(si);
				ss->copyFromEigen3(m->state, V.row(j));
				m->motion_index = j;
				m->forest_index = i;
				nn->add(m);
				all_motions.emplace_back(m);
			}
			if (verbose) {
				std::cerr << i + 1 << " / " << ex_graph_v_.size() << std::endl;
			}
		}
		// Try to connect forests
		std::vector<Motion*> nmotions;
		for (size_t i = 0; i < all_motions.size(); i++) {
			auto m = all_motions[i];
			nn->nearestK(m, KNN, nmotions);
			for (auto nn: nmotions) {
				if (m->forest_index == nn->forest_index)
					continue;
				if (!si->checkMotion(m->state, nn->state))
					continue;
				Eigen::Vector4i e;
				e << m->forest_index, m->motion_index,
				     nn->forest_index, nn->motion_index;
				edges.emplace_back(e);
			}
			// pc: percent
			int pc = i / (all_motions.size() / 100);
			if (verbose && last_pc < pc) {
				if (pc > 0)
					std::cerr << "\r";
				std::cerr << pc << "%";
				last_pc = pc;
			}
			std::cerr << std::endl;
		}/*}}}*/
	} else if (version == 1) {
		/*
		 * For all N blooming trees, we one KNN DS each of
		 * them.
		 */
		ex_knn_.clear();/*{{{*/
		for (size_t i = 0; i < ex_graph_v_.size(); i++) {
			const auto& V = ex_graph_v_[i];
			auto nn = createKNNForRDT(real_planner.get());
			for (int j = 0; j < V.rows(); j++) {
				auto m = new Motion(si);
				ss->copyFromEigen3(m->state, V.row(j));
				m->motion_index = j;
				m->forest_index = i;
				nn->add(m);
				all_motions.emplace_back(m);
			}
			ex_knn_.emplace_back(std::move(nn));
			if (verbose) {
				std::cerr << i + 1 << " / " << ex_graph_v_.size() << std::endl;
			}
		}
		// KNN for each tree
		std::vector<Motion*> nmotions;
		struct MotionWithDistance {
			Motion* motion;
			double distance;
		};
		std::vector<MotionWithDistance> nmotions_all;
		for (size_t i = 0; i < all_motions.size(); i++) {
			auto m = all_motions[i];
			for (int j = 0; j < ex_graph_v_.size(); j++) {
				nmotions.clear();
				if (i == j)
					continue;
				ex_knn_[j]->nearestK(m, KNN, nmotions);
				for (auto nm : nmotions) {
					double d = real_planner->distanceFunction(m, nm);
					nmotions_all.emplace_back(MotionWithDistance{.motion = nm, .distance = d});
				}
			}
			nmotions.clear();
			std::nth_element(nmotions_all.begin(),
					 nmotions_all.begin() + KNN,
					 nmotions_all.end(),
					 [](const MotionWithDistance& lhs, const MotionWithDistance& rhs) {
						return lhs.distance < rhs.distance;
					 });
			for (size_t j = 0; j < KNN; j++) {
				auto nm = nmotions_all[j].motion;
				if (!si->checkMotion(m->state, nm->state))
					continue;
				Eigen::Vector4i e;
				e << m->forest_index, m->motion_index,
				     nm->forest_index, nm->motion_index;
				edges.emplace_back(e);
			}
			// pc: percent
			int pc = i / (all_motions.size() / 100);
			if (verbose && last_pc < pc) {
				std::cerr << pc << "%" << std::endl;
				last_pc = pc;
			}
		}/*}}}*/
	} else if (version == 2) {
		/*
 		 * Similar to Version 1, but for each node in current tree,
		 * Version 2 checks all edges to the remaining
		 * (N-1) KNN DS, while Version 1 only check the nearest K.
		 *
		 * Version 1 and 2 were not designed with multi processing in
		 * mind. Hence N KNN DS were used to avoid recreating KNN DS
		 * for each tree.
		 */
		auto NTree = ex_graph_v_.size();/*{{{*/
		ex_knn_.clear();
		std::vector<int> tree_offset(NTree + 1);
		for (size_t i = 0; i < NTree; i++) {
			const auto& V = ex_graph_v_[i];
			auto nn = createKNNForRDT(real_planner.get());
			tree_offset[i] = all_motions.size();
			for (int j = 0; j < V.rows(); j++) {
				auto m = new Motion(si);
				ss->copyFromEigen3(m->state, V.row(j));
				m->motion_index = j;
				m->forest_index = i;
				nn->add(m);
				all_motions.emplace_back(m);
			}
			ex_knn_.emplace_back(std::move(nn));
			if (verbose) {
				std::cerr << i + 1 << " / " << ex_graph_v_.size() << "\tall_motions size: " << all_motions.size() << std::endl;
			}
		}
		// Ensure tree_offset[tree_id+1] always points to the
		// upper bound.
		tree_offset[NTree] = all_motions.size();
		// Empty subset means select all
		if (subset.size() == 0) {
			subset.resize(NTree);
			for (size_t i = 0; i < NTree; i++)
				subset(i) = i;
		}
		for (int i = 0; i < subset.size(); i++) {
			auto subset_id = subset(i);
			auto qfrom = tree_offset[subset_id];
			auto qto = tree_offset[subset_id+1];
			if (verbose) {
				std::cerr << "Subset: " << i + 1 << " / " << subset.size()
					  << " From " << qfrom << " To " << qto
					  << std::endl;
			}
			int last_pc = -1;
			ssize_t total = (qto - qfrom) * NTree;
			for (int mi = qfrom; mi < qto; mi++) {
				auto m = all_motions[mi];
				for (int k = 0; k < NTree; k++) {
					if (m->forest_index == k)
						continue;
					std::vector<Motion*> nmotions;
					ex_knn_[k]->nearestK(m, KNN, nmotions);
					if (verbose) {
						int pc = (mi - qfrom) * NTree + k;
						pc = pc * 100 / total;
						if (pc != last_pc) {
							std::cerr << "[Subset " << i + 1 << "/" << subset.size() << "] " << pc << "%" << std::endl;
							last_pc = pc;
						}
					}
					for (auto nn: nmotions) {
						if (!si->checkMotion(m->state, nn->state))
							continue;
						Eigen::Vector4i e;
						e << m->forest_index, m->motion_index,
						     nn->forest_index, nn->motion_index;
						edges.emplace_back(e);
					}
				}
			}
		}/*}}}*/
	} else if (version == 3 || version == 4) {
		/*
 		 * Version 3 and 4 are the multi-processed variants of Version 0.
		 *
		 * This algorithm takes one tree ID as input, and output edges
		 * to the remaining trees. Unlike Version 0, the nodes from
		 * input tree were not added to the KNN DS
		 *
		 * Version 4 will further delete nodes from already connected
		 * trees.
		 */
		if (subset.size() != 1)
			throw std::runtime_error("mergeExistingGraph algorithm ver. 3 requies one and only one element in argument subset");
		auto NTree = ex_graph_v_.size();
		std::vector<int> tree_offset(NTree + 1);
		ex_knn_.clear();
		auto nn = createKNNForRDT(real_planner.get());
		ex_knn_.emplace_back(nn);
		int source = subset[0];
		std::unordered_set<int> knn_containing(NTree);

		for (size_t i = 0; i < NTree; i++) {
			const auto& V = ex_graph_v_[i];
			tree_offset[i] = all_motions.size();
			for (int j = 0; j < V.rows(); j++) {
				auto m = new Motion(si);
				ss->copyFromEigen3(m->state, V.row(j));
				m->motion_index = j;
				m->forest_index = i;
				if (i != source)
					nn->add(m);
				all_motions.emplace_back(m);
			}
			if (V.rows() > 0 && i != source)
				knn_containing.insert(i);
		}
		// upper bound.
		tree_offset[NTree] = all_motions.size();
		auto qfrom = tree_offset[source];
		auto qto = tree_offset[source+1];
		if (verbose) {
			std::cerr << "Subset: From " << qfrom << " To " << qto
				  << std::endl;
		}
		int last_pc = -1;
		ssize_t total = qto - qfrom;
		for (int mi = qfrom; mi < qto; mi++) {
			auto m = all_motions[mi];
			std::vector<Motion*> nmotions;
			nn->nearestK(m, KNN, nmotions);
			if (verbose) {
				int pc = mi - qfrom;
				pc = pc * 100 / total;
				if (pc != last_pc) {
					std::cerr << pc << "%" << std::endl;
					last_pc = pc;
				}
			}
			for (auto n: nmotions) {
				if (!si->checkMotion(m->state, n->state))
					continue;
				Eigen::Vector4i e;
				e << m->forest_index, m->motion_index,
				     n->forest_index, n->motion_index;
				edges.emplace_back(e);
				if (version == 4) {
					auto iter = knn_containing.find(n->forest_index);
					if (iter != knn_containing.end()) {
						// Remove the whole tree from
						// the KNN DS
						for (auto t = tree_offset[n->forest_index]; t < tree_offset[n->forest_index + 1]; t++) {
							nn->remove(all_motions[t]);
						}
						knn_containing.erase(iter);
					}
				}
				
			}
		}
	}
	updatePerformanceNumbers(setup);
	// FIXME: all_motions is leaked
	Eigen::MatrixXi ret;
	ret.resize(edges.size(), 4);
	for (size_t i = 0; i < edges.size(); i++)
		ret.row(i) << edges[i](0), edges[i](1), edges[i](2), edges[i](3);
	return ret;
}


Eigen::VectorXi
OmplDriver::validateStates(const Eigen::MatrixXd& qs0)
{
	ompl::app::SE3RigidBodyPlanning setup;
	{
		auto bak = planner_id_;
		planner_id_ = PLANNER_ReRRT;
		configSE3RigidBodyPlanning(setup, false);
		planner_id_ = bak;
	}
	auto generic_planner = setup.getPlanner();
	auto si = generic_planner->getSpaceInformation();
	auto ss = si->getStateSpace();
	Eigen::VectorXi ret;
	ret.setZero(qs0.rows());
	auto m0 = new Motion(si);
	for (int i = 0; i < qs0.rows(); i++) {
		ss->copyFromEigen3(m0->state, qs0.row(i));
		if (si->isValid(m0->state))
			ret(i) = 1;
	}
	delete m0;
	return ret;
}


Eigen::VectorXi
OmplDriver::validateMotionPairs(const Eigen::MatrixXd& qs0,
                                const Eigen::MatrixXd& qs1)
{
	ompl::app::SE3RigidBodyPlanning setup;
	{
		auto bak = planner_id_;
		planner_id_ = PLANNER_ReRRT;
		configSE3RigidBodyPlanning(setup, false);
		planner_id_ = bak;
	}
	auto generic_planner = setup.getPlanner();
	auto si = generic_planner->getSpaceInformation();
	auto ss = si->getStateSpace();

	Eigen::VectorXi ret;
	ret.setZero(qs0.rows());
	auto m0 = new Motion(si);
	auto m1 = new Motion(si);
	for (int i = 0; i < std::min(qs0.rows(), qs1.rows()); i++) {
		ss->copyFromEigen3(m0->state, qs0.row(i));
		ss->copyFromEigen3(m1->state, qs1.row(i));
		if (si->checkMotion(m0->state, m1->state))
			ret(i) = 1;
	}
	delete m0;
	delete m1;
	return ret;
}


void
OmplDriver::configSE3RigidBodyPlanning(ompl::app::SE3RigidBodyPlanning& setup,
                                       bool continuous)
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


void
OmplDriver::updatePerformanceNumbers(ompl::app::SE3RigidBodyPlanning& setup)
{
	auto si = setup.getSpaceInformation();
	auto validator = si->getMotionValidator();
	auto generic_planner = setup.getPlanner();
	auto real_planner = std::dynamic_pointer_cast<ompl::geometric::ReRRT>(generic_planner);
	if (real_planner) {
		auto nn = real_planner->_accessNearestNeighbors();
		latest_pn_.knn_time = nn->getTimeCounter() * 1e-6;
	} else {
		latest_pn_.knn_time = 0.0;
	}
	latest_pn_.motion_check = validator->getCheckedMotionCount();
	latest_pn_.motion_check_time = validator->getMotionCheckTime() * 1e-6;
	latest_pn_.motion_discrete_state_check = validator->getCheckedDiscreteStateCount();
}
