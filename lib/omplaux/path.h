#ifndef OMPLAUX_PATH_H
#define OMPLAUX_PATH_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct Path {
	typedef Eigen::Matrix<float, 4, 4, Eigen::ColMajor> GLMatrix;
	typedef Eigen::Matrix<double, 4, 4, Eigen::ColMajor> GLMatrixd;
	std::vector<Eigen::Vector3d> T;
	std::vector<Eigen::Quaternion<double>> Q;
	//Eigen::aligned_vector<fcl::Transform3d> M;

	void readPath(const std::string& fn)
	{
		std::ifstream fin(fn);
		while (true) {
			double x, y, z;
			fin >> x >> y >> z;
			if (fin.eof())
				break;
			else if (fin.fail())
				throw std::runtime_error("Failure occurs during reading " + fn);
			T.emplace_back(x, y, z);
			double qx, qy, qz, qw;
			// If no rotation is represented with (0,0,0,1)
			// we know it's xyzw sequence because w = cos(alpha/2) = 1 when
			// alpha = 0.
			fin >> qx >> qy >> qz >> qw;
			Q.emplace_back(qw, qx, qy, qz);
		}
#if 0
		for (size_t i = 0; i < T.size(); i++) {
			std::cerr << T[i].transpose() << "\t" << Q[i].vec().transpose() << " " << Q[i].w() << std::endl;
		}
#endif
		std::cerr << "T size: " << T.size() << std::endl;
	}

	GLMatrixd interpolate(const Geo& robot, double t)
	{
		int i = std::floor(t);
		double c = t - double(i);
		int from = i % T.size();
		int to = (i + 1) % T.size();

		GLMatrixd ret;
		ret.setIdentity();
		// Translate robot's center to origin
		ret.block<3,1>(0, 3) = -robot.center;
		// Rotation w.r.t. the translated origin, i.e. robot's center
		Eigen::Quaternion<double> Qfrom = Q[from];
		Eigen::Quaternion<double> Qinterp = Qfrom.slerp(c, Q[to]);
		auto rotmat = Qinterp.toRotationMatrix();
		ret.block<3,3>(0,0) = rotmat;
		ret.block<3,1>(0,3) = rotmat * (-robot.center);
		// Translation
		Eigen::Vector3d translate = T[from] * (1 - c) + T[to] * c;
		GLMatrixd trback;
		trback.setIdentity();
		trback.block<3,1>(0, 3) = translate;
		ret = trback * ret; // Trback * Rot * Tr2Origin
		return ret;
	}

	template<typename FLOAT>
	static GLMatrixd stateToMatrix(const Eigen::Matrix<FLOAT, 6, 1>& state,
			Eigen::Vector3d center = Eigen::Vector3d::Zero()
			)
	{
		Eigen::Transform<FLOAT, 3, Eigen::AffineCompact> tr;
		tr.setIdentity();
		tr.translate(-center);
		tr.rotate(Eigen::AngleAxisd(state(3), Eigen::Vector3d::UnitX()));
		tr.rotate(Eigen::AngleAxisd(state(4), Eigen::Vector3d::UnitY()));
		tr.rotate(Eigen::AngleAxisd(state(5), Eigen::Vector3d::UnitZ()));
		// Note, the transliation component in state represents the
		// translation w.r.t the original geometry.
		Eigen::Vector3d vec(state(0), state(1), state(2));
		tr.translate(vec);
		GLMatrixd ret;
		ret.setIdentity();
		ret.block<3, 4>(0, 0) = tr.matrix();
		return ret;
	}

	// stateToPath only needs to convert Euler angles to quaternion
	// Path::interpolate is responsible for calculating the correct tranform from the path data.
	template<typename FLOAT>
	static Eigen::Matrix<FLOAT, 7, 1> stateToPath(const Eigen::Matrix<FLOAT, 6, 1>& state)
	{
		Eigen::Matrix<FLOAT, 7, 1>  ret;
		Eigen::Quaternion<FLOAT> Q;
		using Vector3F = Eigen::Matrix<FLOAT, 3, 1>;

		Eigen::AngleAxis<FLOAT> r(state(3), Vector3F::UnitX());
		Eigen::AngleAxis<FLOAT> p(state(4), Vector3F::UnitY());
		Eigen::AngleAxis<FLOAT> y(state(5), Vector3F::UnitZ());
		Q = r * p * y;

		ret << state(0), state(1), state(2),
		       Q.x(), Q.y(), Q.z(), Q.w();
		return ret;
	}

	template<typename FLOAT = double>
	Eigen::Matrix<FLOAT, 6, 1> pathToState(size_t idx)
	{
		Eigen::Matrix<FLOAT, 6, 1> state;

		const auto& q = Q[idx];
		// The following converting code is copied from Wikipedia
		double ysqr = q.y() * q.y();

		// roll (x-axis rotation)
		double t0 = +2.0f * (q.w() * q.x() + q.y() * q.z());
		double t1 = +1.0f - 2.0f * (q.x() * q.x() + ysqr);
		double roll = std::atan2(t0, t1);

		// pitch (y-axis rotation)
		double t2 = +2.0f * (q.w() * q.y() - q.z() * q.x());
		t2 = t2 > 1.0f ? 1.0f : t2;
		t2 = t2 < -1.0f ? -1.0f : t2;
		double pitch = std::asin(t2); // asin returns [-pi/2, pi/2]

		// yaw (z-axis rotation)
		double t3 = +2.0f * (q.w() * q.z() + q.x() *q.y());
		double t4 = +1.0f - 2.0f * (ysqr + q.z() * q.z());  
		double yaw = std::atan2(t3, t4);

		state << T[idx](0), T[idx](1), T[idx](2), roll, pitch, yaw;

		return state;
	}

	// TODO: check consistency b/w stateToMatrix and matrixToState
	// Note: this also needn't robot center because it takes
	// the output of Path::interpolate.
	static Eigen::Matrix<double, 6, 1> matrixToState(const GLMatrixd& trmat)
	{
		Eigen::Matrix<double, 6, 1> ret;
		Eigen::Vector3d ea = trmat.block<3,3>(0,0).eulerAngles(0,1,2);
		ret << trmat(0,3), trmat(1,3), trmat(2,3), ea(0), ea(1), ea(2);
		return ret;
	}
};

#endif
