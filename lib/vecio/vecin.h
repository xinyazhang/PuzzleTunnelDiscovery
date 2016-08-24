#ifndef VECIN_H
#define VECIN_H

#include <string>
#include <Eigen/Core>

namespace vecio { 
	void text_read(const std::string& fn, Eigen::VectorXd& );
};

#endif
