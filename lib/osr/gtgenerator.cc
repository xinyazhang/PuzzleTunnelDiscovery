#include "gtgenerator.h"
#include "osr_state.h"
#include "osr_render.h"
#include <vecio/matio.h>
#include <ompl/datastructures/NearestNeighborsGNAT.h>
#include <fstream>

namespace osr {

constexpr int kNearestFromInitState = 8;
constexpr double kEpsilonDistance = 1e-8;

using Edge = GTGenerator::Edge;
using Vertex = GTGenerator::Vertex;
using NNVertex = GTGenerator::NNVertex;

std::istream& operator >> (std::istream& fin, Edge& e)
{
	return fin >> e.first >> e.second;
}

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
		for (const auto& vp : V_)
			nn_->add(vp.get());
	}
};

GTGenerator::GTGenerator(Renderer& r)
	:r_(r)
{
}

GTGenerator::~GTGenerator()
{
}

void GTGenerator::loadRoadMapFile(const std::string& fn)
{
	loadRoadMap(std::ifstream(fn));
}

void GTGenerator::loadRoadMap(std::istream&& fin)
{
	while (!fin.eof()) {
		char type;
		fin >> type;
		if (fin.eof())
			break;
		if (type == 'v') {
			auto *nv = new Vertex;
			vecio::read(fin, nv->state);
			knn_->add(nv);
		} else if (type == 'e') {
			Edge e;
			fin >> e;
			knn_->add(e);
		} else if (type == 'p') {
			Edge e;
			fin >> e;
			if (verifyEdge(e))
				knn_->add(e);
		}
	}

	knn_->build();
	initValue();
}

bool GTGenerator::verifyEdge(const GTGenerator::Edge& e) const
{
	auto s0 = r_.translateToUnitState(knn_->V_[e.first]->state);
	auto s1 = r_.translateToUnitState(knn_->V_[e.second]->state);
	return std::get<1>(r_.transitStateTo(s0, s1, verify_magnitude));
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

void GTGenerator::initValue()
{
	int NV = int(knn_->V_.size());
	dist_values_ = Eigen::VectorXf(NV, -1);
	Eigen::VectorXi done_marks(NV, 0);

	dist_values_[getGoalStateIndex()] = getGoalStateReward();
	auto cmp = [this](NNVertex lhs, NNVertex rhs) -> bool
	{
		return dist_values_[lhs->index] > dist_values_[rhs->index];
	};
	std::priority_queue<NNVertex, std::vector<NNVertex>, decltype(cmp)> Q(cmp);

	for (int i = 0; i < int(knn_->V_.size()); i++) {
		auto nv = knn_->V_[i].get();
		if (r_.isDisentangled(nv->state)) {
			Q.push(nv);
			dist_values_(i) = 0.0;
		}
	}

	while (!Q.empty()) {
		auto tip = Q.top();
		Q.pop();
		auto index = tip->index;
		if (done_marks(index))
			continue;
		done_marks(index) = 1;
		auto tip_value = dist_values_(tip->index);
		for (auto adj_index : tip->adjs) {
			auto adj = knn_->V_[adj_index].get();
			auto steps = estimateSteps(tip, adj);
			bool do_update = dist_values_(adj_index) < 0 or
				         dist_values_(adj_index) > tip_value + steps;
			if (do_update) {
				dist_values_(adj_index) = tip_value + steps;
				adj->next = tip->index;
				Q.push(adj);
			}
		}
	}
}

float GTGenerator::estimateSteps(NNVertex from, NNVertex to) const
{
	return KNN::distance(from, to) / rl_stepping_size;
}

std::tuple<ArrayOfStates, Eigen::VectorXi>
GTGenerator::generateGTPath(const StateVector& init_state,
                            int max_steps)
{
	Vertex init_vertex;
	init_vertex.index = -1;
	init_vertex.state = init_state;

	std::vector<NNVertex> neighbors;
	knn_->nn_->nearestK(&init_vertex, kNearestFromInitState, neighbors);

	NNVertex next = nullptr;
	auto s0 = r_.translateToUnitState(init_state);
	for (const auto neigh : neighbors) {
		auto s1 = r_.translateToUnitState(neigh->state);
		if (std::get<1>(r_.transitStateTo(s0, s1, verify_magnitude))) {
			next = neigh;
			break;
		}
	}
	init_vertex.next = next->index;
	// TODO: cannot find neighbour
	assert(next != nullptr);
	std::vector<StateVector> states;
	std::vector<int> actions;

	/*
	 * current means the a vertex that's close to the current_state
	 */
	NNVertex current = &init_vertex;
	StateVector current_state = init_state;
	while (current->index != getGoalStateIndex()) {
		next = knn_->V_[current->index].get();
		current_state = castTrajectory(current_state,
				next->state,
				states,
				actions);
		current = next;
	}
	ArrayOfStates ret_states;
	ret_states.resize(states.size(), Eigen::NoChange);
	for (size_t i = 0; i < states.size(); i++)
		ret_states.row(i) = states[i];
	Eigen::VectorXi ret_actions;
	ret_actions.resize(actions.size(), Eigen::NoChange);
	for (size_t i = 0; i < actions.size(); i++)
		ret_actions(i) = actions[i];
	return std::make_tuple(ret_states, ret_actions);
}

StateVector
GTGenerator::castTrajectory(const StateVector& from,
			    const StateVector& to,
			    std::vector<StateVector>& states,
			    std::vector<int>& actions)
{
	StateVector current = from;
	while (true) {
		Eigen::VectorXd dists(kTotalNumberOfActions);
		ArrayOfStates nexts;
		nexts.resize(kTotalNumberOfActions, Eigen::NoChange);
		for (int i = 0; i < kTotalNumberOfActions; i++) {
			auto tup = r_.transitState(current, i, rl_stepping_size, verify_magnitude);
			if (std::get<2>(tup) == 0.0) {
				dists(i) = 1000.0;
				continue;
			}
			dists(i) = distance(std::get<0>(tup), to);
			nexts.row(i) = std::get<0>(tup);
		}
		int action;
		dists.minCoeff(&action);
		states.emplace_back(nexts.row(action));
		actions.emplace_back(action);
	}
	return current;
}

}
