#include "gtgenerator.h"
#include "osr_state.h"
#include "osr_render.h"
#include <vecio/matio.h>
#include <ompl/datastructures/NearestNeighborsGNAT.h>
#include <fstream>
#include <atomic>

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

GTGenerator::GTGenerator(Renderer& r)
	:r_(r), knn_(new KNN)
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
		#pragma omp parallel for
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

	Progress prog("Dijkstra", knn_->V_.size());
	for (int i = 0; i < int(knn_->V_.size()); i++) {
		auto nv = knn_->V_[i].get();
		if (r_.isDisentangled(nv->state)) {
			Q.push(nv);
			dist_values_(i) = 0.0;
			prog.increase();
		}
	}

	while (!Q.empty()) {
		auto tip = Q.top();
		Q.pop();
		auto index = tip->index;
		if (done_marks(index))
			continue;
		done_marks(index) = 1;
		prog.increase();
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
	auto s0 = r_.translateToUnitState(knn_->V_[e.first]->state);
	auto s1 = r_.translateToUnitState(knn_->V_[e.second]->state);
	std::cerr << "state distance " << distance(s0, s1) << std::endl;
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

float GTGenerator::estimateSteps(NNVertex from, NNVertex to) const
{
	return KNN::distance(from, to) / rl_stepping_size;
}

std::tuple<ArrayOfStates, Eigen::VectorXi, bool>
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
	bool terminated = true;
	while (current->index != getGoalStateIndex()) {
		next = knn_->V_[current->index].get();
		current_state = castTrajectory(current_state,
				next->state,
				states,
				actions);
		current = next;
		if (actions.size() > max_steps) {
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
