#ifndef VECOUT_H
#define VECOUT_H

#include <string>
#include <Eigen/Core>

namespace vecio { 
	void text_write(const std::string& fn, const Eigen::VectorXd& );
};

#endif

