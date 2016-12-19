#ifndef NAIVE_RENDERER_H
#define NAIVE_RENDERER_H

#include <Eigen/Core>

class NaiveRenderer {
public:
	virtual void addSplit(const Eigen::VectorXd& center, 
			      const Eigen::VectorXd& mins,
			      const Eigen::VectorXd& maxs) {}
	virtual void addCertain(const Eigen::VectorXd& center, 
				const Eigen::VectorXd& mins,
				const Eigen::VectorXd& maxs) {}
};

class Naive2DRenderer : public NaiveRenderer {
public:
	virtual void addSplit(const Eigen::VectorXd& center, 
			      const Eigen::VectorXd& mins,
			      const Eigen::VectorXd& maxs);
	virtual void addCertain(const Eigen::VectorXd& center, 
				const Eigen::VectorXd& mins,
				const Eigen::VectorXd& maxs);

	void init();
	void launch_worker(std::function<int(NaiveRenderer*)>);
	int run();
};

#endif
