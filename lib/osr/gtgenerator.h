/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
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

class UnitWorld;
class Progress;

class GTGenerator {
	struct KNN;
public:
	GTGenerator(UnitWorld&);
	~GTGenerator();

	void loadRoadMapFile(const std::string&);
	void loadRoadMap(std::istream&&);
	void saveVerifiedRoadMapFile(const std::string& fn);
	void initKNN(); // Initialize internal KNN structure
	void initKNNInBatch(); // Initialize internal KNN structure with its std::vector interface
	void initGT();  // Initialize Ground Truth Distance

	void installGTData(const ArrayOfStates& vertices,
	                   const Eigen::Matrix<int, -1, 2>& edges,
	                   const Eigen::VectorXf& gt_distance,
	                   const Eigen::VectorXi& gt_next);
	std::tuple<ArrayOfStates,
	           Eigen::Matrix<int, -1, 2>,
	           Eigen::VectorXf,
	           Eigen::VectorXi>
	extractGTData() const;

	/*
	 * generate states, and corresponding actions for training
	 * 
	 * implicity argument: rl_stepping_size for magnitude of actions.
	 */
	std::tuple<ArrayOfStates, Eigen::VectorXi, bool>
	generateGTPath(const StateVector& init_state,
	               int max_steps,
	               bool for_rl = true);

	std::tuple<ArrayOfStates, Eigen::VectorXi>
	projectTrajectory(const StateVector& from,
	                  const StateVector& to,
	                  int max_steps = -1,
	                  bool in_unit = true);

	/*
	 * Returns array of translation, array of axis angle rotations, and
	 * boolean for if path terminated.
	 * 
	 * All actions are in UnitWorld
	 */
	std::tuple<ArrayOfTrans, ArrayOfAA, Eigen::VectorXd, bool>
	castPathToContActionsInUW(const ArrayOfStates& path,
	                          bool path_is_verified = false);

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
	UnitWorld& uw_;
	std::unique_ptr<KNN> knn_;
	Eigen::VectorXf dist_values_;

	bool verifyEdge(const Edge& e) const;
	int getGoalStateIndex() const;
	float getGoalStateReward() const;

	float estimateSteps(NNVertex from, NNVertex to) const;

	/*
	 * Given the starting state "from", and LERP it to "to"
	 * The intermediate states and actions are pushed into corresponding
	 * vectors.
	 * 
	 * Returns the final state since it is usually not possible to get
	 * "to" accuractly.
	 * 
	 * choose_in_unit: use the closest point in unit world (true) or
	 *                 original world (false)
	 */
	StateVector castTrajectory(const StateVector& from,
	                           const StateVector& to,
	                           std::vector<StateVector>& states,
	                           std::vector<int>& actions,
	                           int max_steps,
	                           Progress* = nullptr,
	                           bool for_rl = true,
	                           bool choose_in_unit = true);
};

}

#endif
