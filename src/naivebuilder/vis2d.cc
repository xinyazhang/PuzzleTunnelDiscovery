#include "vis2d.h"

NaiveRenderer* NaiveVisualizer::renderer_;
int NaiveVisualizer::aggpath_token;

std::ostream& operator<<(std::ostream& fout, const std::vector<Eigen::VectorXd>& milestones)
{
	for(const auto& m : milestones) {
		fout << m.transpose() << std::endl;
	}
	return fout;
}
