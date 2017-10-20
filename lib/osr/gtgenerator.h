/*
 * Note: Logically GT Generator should not be a part of Off-Screen Rendering
 *       Module, but for simplicity it is here, and hence OSR is used rather
 *       than OFF_SCREEN_RENDERING here.
 * 
 */
#ifndef OSR_GT_GENERATOR_H
#define OSR_GT_GENERATOR_H

#include <istream>
#include <utility>
#include <vector>
#include <memory>
#include "osr_state.h"

namespace osr {

class Renderer;

class GTGenerator {
	struct KNN;
public:
	GTGenerator(Renderer&);
	~GTGenerator();

	void loadRoadMapFile(const std::string&);
	void loadRoadMap(std::istream&&);

	/*
	 * generate states, and corresponding actions for training
	 * 
	 * implicity argument: rl_stepping_size for magnitude of actions.
	 */
	std::tuple<ArrayOfStates, Eigen::VectorXi>
	generateGTPath(const StateVector& init_state,
	               int max_steps);

	struct Vertex {
		int index;
		StateVector state;
		std::vector<int> adjs;
		int next = -1;
	};
	using NNVertex = Vertex*; // Note: Pointer, not actual object.
	using Edge = std::pair<int, int>;

	double verify_magnitude = 0.00125;
	float gamma = 0.99; // Reward decay factor in RL.

	/*
	 * This should be the same as the 'magnitudes' argument in
	 * Renderer::transitStateTo
	 * 
	 * TODO: unify this with Renderer::transitState
	 */
	double rl_stepping_size;
private:
	Renderer& r_;
	std::unique_ptr<KNN> knn_;
	Eigen::VectorXf dist_values_;

	bool verifyEdge(const Edge& e) const;
	int getGoalStateIndex() const;
	float getGoalStateReward() const;

	void initValue();
	float estimateSteps(NNVertex from, NNVertex to) const;

	/*
	 * Given the starting state "from", and LERP it to "to"
	 * The intermediate states and actions are pushed into corresponding
	 * vectors.
	 * 
	 * Returns the final state since it is usually not possible to get
	 * "to" accuractly.
	 */
	StateVector castTrajectory(const StateVector& from,
	                           const StateVector& to,
	                           std::vector<StateVector>& states,
	                           std::vector<int>& actions);
};

}

#endif
