#ifndef NAIVE_RENDERER_H
#define NAIVE_RENDERER_H

#include <Eigen/Core>
#include <memory>

struct Geo;

class NaiveRenderer {
public:
	virtual void addSplit(const Eigen::VectorXd& center, 
			      const Eigen::VectorXd& mins,
			      const Eigen::VectorXd& maxs) {}
	virtual void addCertain(const Eigen::VectorXd& center, 
				const Eigen::VectorXd& mins,
				const Eigen::VectorXd& maxs,
				bool isfree) {}
	virtual void addLine(const Eigen::MatrixXd& LS) {}
	virtual void setEnv(const Geo*) {};
	virtual void workerReady() {};
};

class Naive2DRenderer : public NaiveRenderer {
public:
	Naive2DRenderer();
	~Naive2DRenderer();

	virtual void addSplit(const Eigen::VectorXd& center, 
			      const Eigen::VectorXd& mins,
			      const Eigen::VectorXd& maxs) override;
	virtual void addCertain(const Eigen::VectorXd& center, 
				const Eigen::VectorXd& mins,
				const Eigen::VectorXd& maxs,
				bool isfree) override;
	virtual void addLine(const Eigen::MatrixXd& LS) override;

	void init();
	void launch_worker(std::function<int(NaiveRenderer*)>);
	virtual void setEnv(const Geo*) override;
	virtual void workerReady() override;

	int run();
private:
	struct Private;
	std::unique_ptr<Private> p_;
};

#endif
