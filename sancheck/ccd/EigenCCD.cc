#include "EigenCCD.h"
#include <igl/readOBJ.h>
#include <igl/ray_mesh_intersect.h>
#include <omplaux/path.h>
#include <Eigen/Geometry>

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
	Eigen::VectorXd geocenter;
	Eigen::Transform<double, 3, Eigen::Affine> tf;
	Eigen::Quaternion<double> rot, inv_rot;
	Eigen::Vector3d tr;
public:
	EigenCCDImpl(const std::string& fn)
	{
		igl::readOBJ(fn, V, F);
		geocenter = V.colwise().mean();
	}

	virtual void support(const ccd_vec3_t *indir, ccd_vec3_t *outvec) override
	{
		Eigen::Vector3d dir;
		dir << *indir;
		dir = inv_rot * dir;

		Eigen::Vector3d vec { Eigen::Vector3d::Zero() };
		std::vector<igl::Hit> hits;
		igl::ray_mesh_intersect(geocenter, dir, V, F, hits);
		for (const auto& hit : hits) {
			if (hit.t > 0) {
				vec = dir * hit.t;
				break;
			}
		}
		std::cerr << "Support vec for " << this << " and direction: " << dir.transpose() << " is: " << vec.transpose() << std::endl;
		*outvec << vec;
	}

	virtual void center(ccd_vec3_t *outvec) override
	{
		*outvec << geocenter;
	}

	virtual void setTransform(const State& state, const Eigen::Vector3d& rot_center = Eigen::Vector3d::Zero()) override
	{
		auto matd = Path::stateToMatrix(state, rot_center);
		std::cerr << matd << std::endl;
		tf.setIdentity();
		tf = matd.block<3,4>(0,0);
		rot = tf.linear();
		inv_rot = rot.inverse();
		tr = tf.translation();
	}
};

std::unique_ptr<EigenCCD> EigenCCD::create(const std::string& fn)
{
	return std::make_unique<EigenCCDImpl>(fn);
}
