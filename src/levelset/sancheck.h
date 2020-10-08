#include <Eigen/Core>

void san_check(
	const Eigen::MatrixXf& IV,
	const Eigen::MatrixXi& IF,
	const Eigen::MatrixXf& OV,
	const Eigen::MatrixXi& OF,
	double expected_distance
	);
