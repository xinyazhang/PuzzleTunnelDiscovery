/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OMPLAUX_BVH_HELPER_H
#define OMPLAUX_BVH_HELPER_H

template<typename BVHModel, typename Splitter, typename VType, typename FType>
void initBVH(BVHModel &bvh,
		Splitter *splitter,
		const VType& V,
		const FType& F)
{
	bvh.bv_splitter.reset(splitter);
	bvh.beginModel();
	std::vector<Eigen::Vector3d> Vs(V.rows());
	std::vector<fcl::Triangle> Fs(F.rows());
	for (int i = 0; i < V.rows(); i++)
		Vs[i] = V.row(i);
	for (int i = 0; i < F.rows(); i++) {
		const auto& f = F.row(i);
		Fs[i] = fcl::Triangle(f(0), f(1), f(2));
	}
	bvh.addSubModel(Vs, Fs);
	bvh.endModel();
}

#endif
