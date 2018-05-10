#include "gtgenerator.h"
#include "osr_state.h"
#include "unit_world.h"
#include <vecio/matio.h>
#include <ompl/datastructures/NearestNeighborsGNAT.h>
#include <fstream>
#include <atomic>

namespace osr {

constexpr int kNearestFromInitState = 8;
constexpr double kEpsilonDistance = 1e-8;
constexpr int kGraphNextFlagNoValue = -1;
constexpr int kGraphNextFlagFinalState = -2;

using Edge = GTGenerator::Edge;
using Vertex = GTGenerator::Vertex;
using NNVertex = GTGenerator::NNVertex;

std::istream& operator >> (std::istream& fin, Edge& e)
{
	return fin >> e.first >> e.second;
}

class Progress {
private:
	size_t total_;
	std::atomic<size_t> counter_;
	const char* task_;
public:
	Progress(const char* task = "Progress", size_t total = 0)
		:task_(task), total_(total), counter_(0)
	{
	}

	void increase()
	{
		auto counter = (++counter_);
		bool show = false;
		if (total_ > 0 && total_ < 10) {
			show = true;
		} else if (total_ >= 10 && counter % (total_/10) == 0) {
			show = true;
		} else if (counter % (100 * 1000) == 0) {
			show = true;
		}
		if (show) {
			std::cerr << "[" << task_ << "] " << counter;
			if (total_ > 0) {
				std::cerr << " / " << total_;
			}
			std::cerr << std::endl;
		}
	}
};

struct GTGenerator::KNN {
	std::vector<std::unique_ptr<Vertex>> V_;
	std::vector<Edge> E_;
	std::unique_ptr<ompl::NearestNeighborsGNAT<Vertex*>> nn_;

	static double distance(const NNVertex& plhv, const NNVertex& prhv)
	{
		return osr::distance(plhv->state, prhv->state);
	}

	KNN()
	{
		// Question: shall we change the default argument like this?
		nn_.reset(new ompl::NearestNeighborsGNAT<NNVertex>(12, 8 ,16));
		nn_->setDistanceFunction(distance);
	}

	~KNN()
	{
		nn_.reset();
	}

	void add(NNVertex nv)
	{
		nv->index = int(V_.size());
		V_.emplace_back(nv);
	}

	void add(const Edge& e)
	{
		E_.emplace_back(e);
		auto from = e.first;
		auto to = e.second;
		V_[from]->adjs.emplace_back(to);
		V_[to]->adjs.emplace_back(from);
	}

	void build()
	{
		Progress prog("KNN", V_.size());
		for (const auto& vp : V_) {
			nn_->add(vp.get());
			prog.increase();
		}
	}

	void batch_build()
	{
		std::cerr << "Building KNN...";
		std::vector<NNVertex> batch_input(V_.size());
		for (size_t i = 0; i < batch_input.size(); i++)
			batch_input[i] = V_[i].get();
		nn_->add(batch_input);
		std::cerr << "Done\n";
	}

	void dumpTo(std::ostream&& fout) const
	{
		fout.precision(17);
		for (const auto& nv : V_) {
			fout << "v";
			for (int i = 0; i < nv->state.rows(); i++)
				fout << " " << nv->state(i);
			fout << std::endl;
		}
		for (const auto& e : E_) {
			fout << "e " << e.first << " " << e.second << std::endl;
		}
	}
};

GTGenerator::GTGenerator(UnitWorld& uw)
	:uw_(uw), knn_(new KNN)
{
}

GTGenerator::~GTGenerator()
{
}

void GTGenerator::loadRoadMapFile(const std::string& fn)
{
	loadRoadMap(std::ifstream(fn));
}

void GTGenerator::saveVerifiedRoadMapFile(const std::string& fn)
{
	knn_->dumpTo(std::ofstream(fn));
}

void GTGenerator::loadRoadMap(std::istream&& fin)
{
	size_t loc = 0;
	Progress prog("Reading");
	std::string typestr;
	std::vector<Edge> pending;
	while (!fin.eof()) {
		fin >> typestr;
		char type = typestr[0];
		// std::cerr << typestr << std::endl;
		if (fin.eof())
			break;
		if (type == 'v') {
			auto *nv = new Vertex;
			// std::cerr << nv->state.rows() << ' ' << nv->state.cols() << std::endl;
			vecio::read(fin, nv->state);
			// std::cerr << nv->state.transpose() << std::endl;
			knn_->add(nv);
		} else if (type == 'e') {
			Edge e;
			fin >> e;
			knn_->add(e);
		} else if (type == 'p') {
			Edge e;
			fin >> e;
			pending.emplace_back(e);
		}
		prog.increase();
#if 0
		loc++;
		if (loc % 100000 == 0) {
			std::cerr << "[Reading] " << loc << " lines" << std::endl;
		}
#endif
	}
	Eigen::VectorXi pending_passed;
	pending_passed.setZero(pending.size());
	Progress evprog("Edge Verification", pending.size());

	{
		// #pragma omp parallel for
		for (int i = 0; i < pending_passed.size(); i++) {
			const auto& e = pending[i];
			if (verifyEdge(e)) {
				pending_passed(i) = 1;
			}
			evprog.increase();
		}
	}

	for (int i = 0; i < pending_passed.size(); i++) {
		if (pending_passed(i)) {
			const auto& e = pending[i];
			knn_->add(e);
		}
	}
}

void GTGenerator::initKNN()
{
	knn_->build();
}

void GTGenerator::initKNNInBatch()
{
	knn_->batch_build();
}

void GTGenerator::initGT()
{
	int NV = int(knn_->V_.size());
	dist_values_ = Eigen::VectorXf::Constant(NV, -1);
	// std::cerr << dist_values_ << std::endl;
	Eigen::VectorXi done_marks = Eigen::VectorXi::Constant(NV, 0);

	dist_values_[getGoalStateIndex()] = getGoalStateReward();
	auto cmp = [this](NNVertex lhs, NNVertex rhs) -> bool
	{
		return dist_values_[lhs->index] > dist_values_[rhs->index];
	};
	std::priority_queue<NNVertex, std::vector<NNVertex>, decltype(cmp)> Q(cmp);

#if 1
	Progress initprog("FindBoundary", knn_->V_.size());
	ArrayOfStates unit_states;
	unit_states.resize(knn_->V_.size(), Eigen::NoChange);
	{
// #pragma omp parallel for
		for (int i = 0; i < unit_states.rows(); i++) {
			auto nv = knn_->V_[i].get();
			auto unit_state = uw_.translateToUnitState(nv->state);
			unit_states.row(i) = unit_state;
			if (uw_.isDisentangled(unit_state)) {
				dist_values_(i) = 0.0;
				nv->next = kGraphNextFlagFinalState;
			}
			initprog.increase();
		}
	}
	for (int i = 0; i < unit_states.rows(); i++)
		if (dist_values_(i) == 0.0)
			Q.push(knn_->V_[i].get());
	std::cerr << "Total bounday states: " << Q.size() << std::endl;
#else
	knn_->V_[getGoalStateIndex()]->next = kGraphNextFlagFinalState;
	Q.push(knn_->V_[getGoalStateIndex()].get());
#endif

	Progress prog("Dijkstra", knn_->V_.size());
	while (!Q.empty()) {
		auto tip = Q.top();
		Q.pop();
		auto index = tip->index;
		if (done_marks(index))
			continue;
		std::cerr << "non-dup tip distance: " << dist_values_(tip->index) << std::endl;
		done_marks(index) = 1;
		prog.increase();
		auto tip_value = dist_values_(tip->index);
		for (auto adj_index : tip->adjs) {
			auto adj = knn_->V_[adj_index].get();
			// auto steps = estimateSteps(tip, adj);
			auto steps = distance(unit_states.row(tip->index),
					      unit_states.row(adj_index));
			bool do_update = dist_values_(adj_index) < 0 or
				         dist_values_(adj_index) > tip_value + steps;
			if (do_update) {
				dist_values_(adj_index) = tip_value + steps;
				adj->next = tip->index;
				Q.push(adj);
				std::cerr << tip->index << " -> "
				          << adj->next << " : "
					  << dist_values_(adj_index)
					  << std::endl;
			}
		}
	}
} 

void GTGenerator::installGTData(const ArrayOfStates& vertices,
	                        const Eigen::Matrix<int, -1, 2>& edges,
	                        const Eigen::VectorXf& gt_distance,
	                        const Eigen::VectorXi& gt_next)
{
	knn_->V_.resize(vertices.rows());
	for(int i = 0; i < vertices.rows(); i++) {
		knn_->V_[i].reset(new Vertex);
		Vertex& v = *knn_->V_[i];
		v.index = i;
		v.state = vertices.row(i);
		v.next = gt_next[i];
	}
	dist_values_ = gt_distance;
	for (int i = 0; i < edges.rows(); i++) {
		Edge e(edges(i,0), edges(i,1));
		knn_->add(e);
	}
}



std::tuple<ArrayOfStates,
	   Eigen::Matrix<int, -1, 2>,
	   Eigen::VectorXf,
	   Eigen::VectorXi>
GTGenerator::extractGTData() const
{
	ArrayOfStates vertices;
	Eigen::Matrix<int, -1, 2> edges;
	Eigen::VectorXf gt_distance;
	Eigen::VectorXi gt_next;

	vertices.resize(knn_->V_.size(), Eigen::NoChange);
	gt_next.resize(knn_->V_.size());
	for(int i = 0; i < vertices.rows(); i++) {
		vertices.row(i) = knn_->V_[i]->state;
		gt_next(i) = knn_->V_[i]->next;
	}
	gt_distance = dist_values_;
	edges.resize(knn_->E_.size(), Eigen::NoChange);
	for (int i = 0; i < edges.rows(); i++) {
		const Edge& e = knn_->E_[i];
		edges.row(i) << e.first, e.second;
	}

	return std::make_tuple(vertices, edges, gt_distance, gt_next);
}

bool GTGenerator::verifyEdge(const GTGenerator::Edge& e) const
{
	auto s0 = uw_.translateToUnitState(knn_->V_[e.first]->state);
	auto s1 = uw_.translateToUnitState(knn_->V_[e.second]->state);
	// std::cerr << "state distance " << distance(s0, s1) << std::endl;
	return std::get<1>(uw_.transitStateTo(s0, s1, verify_magnitude));
}


/*
 * This returns the index of the goal state.
 * 
 * For OMPL, its current implementation always put goal as the second vertex,
 * and hence this function returns a constant.
 * 
 * FIXME: A proper way to establish the convention, rather than
 * assuming/guessing it.
 */
int GTGenerator::getGoalStateIndex() const
{
	return 1;
}

float GTGenerator::getGoalStateReward() const
{
	return 1.0f;
}

float GTGenerator::estimateSteps(NNVertex from, NNVertex to) const
{
	return KNN::distance(from, to) / rl_stepping_size;
}

std::tuple<ArrayOfStates, Eigen::VectorXi, bool>
GTGenerator::generateGTPath(const StateVector& init_state,
                            int max_steps,
                            bool for_rl)
{
	NNVertex next = nullptr;
#if 1
	Vertex init_vertex;
	init_vertex.index = -1;
	init_vertex.state = init_state;
	auto s0 = uw_.translateToUnitState(init_state);

	std::vector<NNVertex> neighbors;
	knn_->nn_->nearestK(&init_vertex, kNearestFromInitState, neighbors);
	std::sort(neighbors.begin(), neighbors.end(),
		[s0, this](const NNVertex& lhs, const NNVertex& rhs) -> bool {
			double ldist = distance(uw_.translateToUnitState(lhs->state), s0);
			double rdist = distance(uw_.translateToUnitState(rhs->state), s0);
			return ldist < rdist;
		}
		 );

	std::cerr << "Finding NN from " << init_state.transpose() << std::endl;
	std::cerr << "  (Checking if Valid) " << uw_.isValid(s0) << std::endl;
	size_t verified_neigh = 0;
	for (const auto neigh : neighbors) {
		auto s1 = uw_.translateToUnitState(neigh->state);
		std::cerr << "\tTrying " << neigh->state.transpose() << std::endl;
		auto tup = uw_.transitStateTo(s0, s1, verify_magnitude);
		std::cerr << "\tEnds at "
			  << uw_.translateFromUnitState(std::get<0>(tup)).transpose()
		          << std::endl;
		std::cerr << "\tTerm " << std::get<1>(tup) << std::endl;
		std::cerr << "\tTau " << std::get<2>(tup) << std::endl;
		if (std::get<1>(tup)) {
			next = neigh;
			break;
		}
		verified_neigh++;
		if (verified_neigh > kNearestFromInitState)
			break;
	}
	if (!next) {
		return std::make_tuple(ArrayOfStates(), Eigen::VectorXi(), false);
	}
	assert(next != nullptr);
	init_vertex.next = next->index;
	/*
	 * current means the a vertex that's close to the current_state
	 */
	NNVertex current = &init_vertex;
#else
	NNVertex current = knn_->V_[0].get();
#endif
	StateVector current_state = init_state;
	// TODO: cannot find neighbour
	std::vector<StateVector> states;
	std::vector<int> actions;
	Progress prog("Taking Actions", max_steps);

	bool terminated = true;
	while (current->index != getGoalStateIndex()) {
		if (current->next == kGraphNextFlagFinalState) {
			break;
		}
		next = knn_->V_[current->next].get();
		current_state = castTrajectory(current_state,
				next->state,
				states,
				actions,
				max_steps,
				&prog,
				for_rl);
		std::cerr << "Trajectory: " << current->index
		          << " -> " << next->index << std::endl;
#if 0
		break;
#endif
		current = next;
		if (actions.size() >= max_steps) {
			terminated = false;
			break;
		}
	}
	ArrayOfStates ret_states;
	ret_states.resize(states.size(), Eigen::NoChange);
	for (size_t i = 0; i < states.size(); i++)
		ret_states.row(i) = states[i];
	Eigen::VectorXi ret_actions;
	ret_actions.resize(actions.size());
	for (size_t i = 0; i < actions.size(); i++)
		ret_actions(i) = actions[i];
	return std::make_tuple(ret_states, ret_actions, terminated);
}

std::tuple<ArrayOfStates, Eigen::VectorXi>
GTGenerator::projectTrajectory(const StateVector& from,
	                       const StateVector& to,
	                       int max_steps,
	                       bool in_unit)
{
	std::vector<StateVector> states;
	std::vector<int> actions;
	castTrajectory(from,
	               to,
	               states,
	               actions,
	               max_steps,
	               nullptr,
	               true,
	               in_unit);
	ArrayOfStates ret_states;
	ret_states.resize(states.size(), Eigen::NoChange);
	for (size_t i = 0; i < states.size(); i++)
		ret_states.row(i) = states[i];
	Eigen::VectorXi ret_actions;
	ret_actions.resize(actions.size());
	for (size_t i = 0; i < actions.size(); i++)
		ret_actions(i) = actions[i];
	return std::make_tuple(ret_states, ret_actions);
}


std::tuple<ArrayOfTrans, ArrayOfAA, Eigen::VectorXd, bool>
GTGenerator::castPathToContActionsInUW(const ArrayOfStates& path,
				       bool path_is_verified)
{
	std::vector<StateTrans> trans_array;
	std::vector<AngleAxisVector> aa_array;
	std::vector<double> mags_array;
	ArrayOfStates unit_states;
	unit_states.resize(path.rows(), Eigen::NoChange);
	Progress uprog("To Unit World", path.rows());
	for (int i = 0; i < path.rows(); i++) {
		unit_states.row(i) = uw_.translateToUnitState(path.row(i));
		uprog.increase();
	}
	bool terminated = true;
	Progress prog("Cont. Actions", path.rows());
	for (int i = 1; i < path.rows(); i++) {
		const StateVector& from = unit_states.row(i-1);
		const StateVector& to = unit_states.row(i);
		double dtau = rl_stepping_size / distance(from, to);
		double end_tau = 1.0;
		if (!path_is_verified) {
			auto tup = uw_.transitStateTo(from, to, verify_magnitude);
			end_tau = std::get<2>(tup);
		}
		double prev_tau = 0;
		double tau = 0;
		bool done = false;
		while (!done) {
			prev_tau = tau;
			tau += dtau;
			if (tau > end_tau) {
				tau = end_tau;
				done = true;
			}
			// std::cerr << tau << std::endl;
			StateVector prev = interpolate(from, to, prev_tau);
			StateVector curr = interpolate(from, to, tau);
			auto diff_tup = differential(prev, curr);
			trans_array.emplace_back(std::get<0>(diff_tup));
			aa_array.emplace_back(std::get<1>(diff_tup));
			mags_array.emplace_back(distance(prev, curr));
		}
		prog.increase();
		if (end_tau < 1.0) {
			terminated = false;
			break;
		}
	}
	ArrayOfTrans ret_trans;
	ret_trans.resize(trans_array.size(), Eigen::NoChange);
	for (size_t i = 0; i < trans_array.size(); i++)
		ret_trans.row(i) = trans_array[i];
	ArrayOfAA ret_aa;
	ret_aa.resize(aa_array.size(), Eigen::NoChange);
	for (size_t i = 0; i < aa_array.size(); i++)
		ret_aa.row(i) = aa_array[i];
	Eigen::VectorXd ret_dists(mags_array.size());
	double distance_from_goal = 0;
	for (int i = int(mags_array.size() - 1); i >= 0; i--) {
		distance_from_goal += mags_array[i];
		ret_dists(i) = distance_from_goal;
	}
	return std::make_tuple(ret_trans, ret_aa, ret_dists, terminated);
}


StateVector
GTGenerator::castTrajectory(const StateVector& from,
			    const StateVector& to,
			    std::vector<StateVector>& states,
			    std::vector<int>& actions,
			    int max_steps,
			    Progress* prog,
			    bool for_rl,
			    bool choose_in_unit)
{
	if (!for_rl) {
		/* W/O CD */
		states.emplace_back(to);
		actions.emplace_back(1);
		return to;
	}
	auto unit_to = uw_.translateToUnitState(to);
	auto unit_current = uw_.translateToUnitState(from);
	while (actions.size() < max_steps or max_steps < 0) {
		auto unit_distance = distance(unit_current, unit_to);
		std::cerr << "CURRENT UNIT DISTANCE "
			  << unit_distance
			  << " VALID? " << uw_.isValid(unit_current)
			  << std::endl;
		if (unit_distance < rl_stepping_size)
			break;
		// dists: store the distance to the goal state
		//        Note the distance is measured in original state
		//        space if choose_in_unit is false.
		Eigen::VectorXd dists(kTotalNumberOfActions);
		Eigen::VectorXd ratios(kTotalNumberOfActions);
		// Eigen::VectorXi valids(kTotalNumberOfActions);
		ArrayOfStates nexts;
		nexts.resize(kTotalNumberOfActions, Eigen::NoChange);
		{
#pragma omp parallel for
		for (int i = 0; i < kTotalNumberOfActions; i++) {
			auto tup = uw_.transitState(unit_current, i,
						   rl_stepping_size,
						   verify_magnitude);
			ratios(i) = std::get<2>(tup);
			// if (ratios(i) == 0.0) {
			if (ratios(i) < 1e-5) {
				dists(i) = 1000.0;
				continue;
			}
			if (choose_in_unit) {
				dists(i) = distance(std::get<0>(tup), unit_to);
			} else {
				auto next = uw_.translateFromUnitState(std::get<0>(tup));
				dists(i) = distance(next, to);
			}
			nexts.row(i) = std::get<0>(tup);
			// valids(i) = uw_.isValid(nexts.row(i));
		}
		}
		int action;
		dists.minCoeff(&action);
		std::cerr << "D: " << dists.transpose() << std::endl;
		std::cerr << "R: " << ratios.transpose() << std::endl;
		// std::cerr << "V: " << valids.transpose() << std::endl;
		if (dists(action) > 100.0) {
			std::cerr << "SAN CHECK 6/2D10: TRAPPED, EVERY ACTION FAILED\n";
			std::cerr << dists << std::endl;
			throw std::runtime_error("SAN CHECK FAILED: TRAPPED");
		}
		double udist;
		if (choose_in_unit)
			udist = dists(action);
		else
			udist = distance(nexts.row(action), unit_to);
		// if (distance(nexts.row(action), to) > distance(current, to)) {
		if (udist > unit_distance) {
			std::cerr << "SAN CHECK 1/1D6: CANNOT CONVERGE INTO STEPPING SIZE\n";
			// std::cerr << "\tCURRENT: " << distance(current, to);
			// std::cerr << "\n\tNEXT: " << distance(nexts.row(action), to) << '\n';
			std::cerr << "\tCURRENT: " << unit_distance;
			std::cerr << "\n\tNEXT: " << dists(action) << '\n';
			throw std::runtime_error("SAN CHECK FAILED: DIVERGED");
		}
		actions.emplace_back(action);
		unit_current = nexts.row(action);
		states.emplace_back(uw_.translateFromUnitState(unit_current));
		if (prog)
			prog->increase();
	}
	return uw_.translateFromUnitState(unit_current); // Note: do not return states.back(), the iteration may not be executed.
}

}
