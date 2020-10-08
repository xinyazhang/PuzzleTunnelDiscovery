#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <string>
#include <Eigen/Core>

class Options {
public:
	Options(int argc, char* argv[]);

	std::istream& get_input_stream();
	void write_geo(const std::string& suffix,
		       const Eigen::MatrixXd& V,
		       const Eigen::MatrixXi& F);

	double margin() const;
private:
	double margin_ = 0.0;
};

#endif
