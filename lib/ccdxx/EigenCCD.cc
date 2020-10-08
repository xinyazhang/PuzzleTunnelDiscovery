/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "EigenCCD.h"
#include <igl/readOBJ.h>
#include <igl/ray_mesh_intersect.h>
#include <omplaux/path.h>

void operator << (ccd_vec3_t& ccdvec, const Eigen::Vector3d& vec)
{
	ccdVec3Set(&ccdvec, vec.x(), vec.y(), vec.z());
}

void operator << (Eigen::Vector3d& vec, const ccd_vec3_t& ccdvec)
{
	vec << ccdVec3X(&ccdvec), ccdVec3Y(&ccdvec), ccdVec3Z(&ccdvec);
}

class EigenCCDImpl : public EigenCCD {
private:
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::Vector3d geocenter;
	Transform3 tf;
	Eigen::Quaternion<double> rot, inv_rot;
	Eigen::Vector3d tr;
public:
	EigenCCDImpl(const std::string& fn)
	{
		igl::readOBJ(fn, V, F);
		geocenter = V.colwise().mean();
	}

	EigenCCDImpl(
			Eigen::MatrixXd&& inV,
			Eigen::MatrixXi&& inF,
			const Eigen::Vector3d* pgeocenter = nullptr
		    ): V(inV), F(inF)
	{
		if (pgeocenter)
			geocenter = *pgeocenter;
		else
			geocenter = V.colwise().mean();
	}

	EigenCCDImpl(
			const Eigen::MatrixXd& inV,
			const Eigen::MatrixXi& inF,
			const Eigen::Vector3d* pgeocenter = nullptr
		    ): V(inV), F(inF)
	{
		if (pgeocenter)
			geocenter = *pgeocenter;
		else
			geocenter = V.colwise().mean();
	}

	virtual void support(const ccd_vec3_t *indir, ccd_vec3_t *outvec) const override
	{
		Eigen::Vector3d dir;
		dir << *indir;
		dir = inv_rot * dir;

		Eigen::Vector3d vec { Eigen::Vector3d::Zero() };
#if 0
		std::vector<igl::Hit> hits;
		igl::ray_mesh_intersect(geocenter, dir, V, F, hits);
		if (!hits.empty())
			vec = dir * hits.back().t;
#endif
		int maxr = 0, maxc = 0;
		Eigen::VectorXd dots = V * dir;
		dots.maxCoeff(&maxr, &maxc);
#if 0
		double maxdot = V.row(0).dot(dir);
		for (int i = 1; i < V.rows(); i++) {
			double dot = V.row(i).dot(dir);
			if (dot > maxdot) {
				maxi = i;
				maxdot = dot;
			}
		}
#endif
		vec = rot * V.row(maxr);
		vec += tr;
		// std::cerr << "Support vec for " << this << " and direction: " << dir.transpose() << " is: " << vec.transpose() << std::endl;
		*outvec << vec;
	}

	virtual void center(ccd_vec3_t *outvec) const override
	{
		*outvec << geocenter;
	}

	virtual void setTransform(const State& state, const Eigen::Vector3d& rot_center = Eigen::Vector3d::Zero()) override
	{
		auto matd = Path::stateToMatrix(state, rot_center);
		// std::cerr << matd << std::endl;
		tf.setIdentity();
		tf = matd.block<3,4>(0,0);
		synctf();
	}

	virtual void setTransform(const Transform3& tfin) override
	{
		tf = tfin;
		synctf();
	}

	void synctf()
	{
		rot = tf.linear();
		inv_rot = rot.inverse();
		tr = tf.translation();
	}
};

std::unique_ptr<EigenCCD> EigenCCD::create(const std::string& fn)
{
	return std::make_unique<EigenCCDImpl>(fn);
}

std::unique_ptr<EigenCCD> EigenCCD::create(
		Eigen::MatrixXd&& V,
		Eigen::MatrixXi&& F,
		const Eigen::Vector3d* pgeocenter
		)
{
	return std::make_unique<EigenCCDImpl>(std::move(V), std::move(F), pgeocenter);
}

std::unique_ptr<EigenCCD> EigenCCD::create(
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			const Eigen::Vector3d* pgeocenter
			)
{
	return std::make_unique<EigenCCDImpl>(V, F, pgeocenter);
}

namespace {

void support(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec)
{
	auto eic = (const EigenCCD *)obj;
	eic->support(dir, vec);
}

void center(const void *obj, ccd_vec3_t *center)
{
	auto eic = (const EigenCCD *)obj;
	eic->center(center);
}

};

bool EigenCCD::penetrate(const EigenCCD* rob, const EigenCCD* env, EigenCCD::PenetrationInfo& info)
{
	int res;
	ccd_t ccd;
	CCD_INIT(&ccd);
	ccd_real_t depth;
	ccd_vec3_t dir, pos;

	ccd.support1 = ::support;
	ccd.support2 = ::support;
	ccd.center1 = ::center;
	ccd.center2 = ::center;
	ccd.max_iterations = 2500;     // maximal number of iterations
	ccd.epa_tolerance = 1e-9;

	res = ccdGJKIntersect(rob, env, &ccd);
	if (!res)
		return false;
	res = ccdMPRPenetration(rob, env, &ccd, &depth, &dir, &pos);
	// int res = ccdGJKPenetration(rob, env, &ccd, &depth, &dir, &pos);
	if (res == 0) {
		info.depth = depth;
		info.dir << dir;
		info.pos << pos;
	}
	return res == 0;
}
