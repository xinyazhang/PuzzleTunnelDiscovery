/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "ompldriver.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

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
	m.attr("PLANNER_RDT"        ) = py::int_(int(PLANNER_ReRRT      ));
	m.attr("PLANNER_RDT_CONNECT") = py::int_(int(PLANNER_RDT_CONNECT));
	m.attr("PLANNER_STRIDE"     ) = py::int_(int(PLANNER_STRIDE     ));
	m.attr("PLANNER_FMT"        ) = py::int_(int(PLANNER_FMT        ));
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
	py::class_<OmplDriver::PerformanceNumbers>(m, "PerformanceNumbers")
		.def(py::init<>())
		.def_readonly("planning_time", &OmplDriver::PerformanceNumbers::planning_time)
		.def_readonly("motion_check", &OmplDriver::PerformanceNumbers::motion_check)
		.def_readonly("motion_check_time", &OmplDriver::PerformanceNumbers::motion_check_time)
		.def_readonly("motion_discrete_state_check", &OmplDriver::PerformanceNumbers::motion_discrete_state_check)
		.def_readonly("knn_query_time", &OmplDriver::PerformanceNumbers::knn_query_time)
		.def_readonly("knn_delete_time", &OmplDriver::PerformanceNumbers::knn_delete_time)
		;
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
		     py::arg("ec_budget") = -1,
		     py::arg("record_compact_tree") = false,
		     py::arg("continuous_motion_validator") = false
		    )
		.def("substitute_state", &OmplDriver::substituteState)
		.def("add_existing_graph", &OmplDriver::addExistingGraph)
		.def("merge_existing_graph", &OmplDriver::mergeExistingGraph,
		     py::arg("knn"),
		     py::arg("verbose") = false,
		     py::arg("version") = 0,
		     py::arg("subset") = Eigen::VectorXi()
		    )
		.def("validate_states", &OmplDriver::validateStates,
		     py::arg("qs0"))
		.def("validate_motion_pairs", &OmplDriver::validateMotionPairs,
		     py::arg("qs0"),
		     py::arg("qs1"))
		.def("set_sample_set", &OmplDriver::setSampleSet)
		.def("set_sample_set_edges", &OmplDriver::setSampleSetEdges,
		     py::arg("QB"),
		     py::arg("QE"),
		     py::arg("QEB")
		    )
		.def("set_sample_set_flags", &OmplDriver::setSampleSetFlags)
		.def("get_sample_set_connectivity", &OmplDriver::getSampleSetConnectivity)
		.def("presample", &OmplDriver::presample)
		.def("get_compact_graph", &OmplDriver::getCompactGraph)
		.def("get_graph_istate_indices", &OmplDriver::getGraphIStateIndices)
		.def("get_graph_gstate_indices", &OmplDriver::getGraphGStateIndices)
		.def_property_readonly("latest_solution", &OmplDriver::getLatestSolution)
		.def_property_readonly("latest_solution_status", &OmplDriver::getLatestSolutionStatus)
		.def_property_readonly("latest_performance_numbers", &OmplDriver::getLatestPerformanceNumbers)
		.def("optimize", &OmplDriver::optimize,
		     py::arg("path"),
		     py::arg("days")
		    )
		;
}
