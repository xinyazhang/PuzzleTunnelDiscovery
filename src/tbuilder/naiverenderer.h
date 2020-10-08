/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
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
	// Returns a token for deletion in the future.
	virtual int addDynamicLine(const Eigen::MatrixXd& LS) { return -1; }
	// Note: undefined behavior for invalid token.
	virtual void removeDynamicLine(int token) { }
	virtual void setEnv(const Geo*) {};
	virtual void workerReady() {};
};

class Naive3DRenderer : public NaiveRenderer {
public:
	Naive3DRenderer();
	~Naive3DRenderer();

	virtual void addSplit(const Eigen::VectorXd& center, 
			      const Eigen::VectorXd& mins,
			      const Eigen::VectorXd& maxs) override;
	virtual void addCertain(const Eigen::VectorXd& center, 
				const Eigen::VectorXd& mins,
				const Eigen::VectorXd& maxs,
				bool isfree) override;
	virtual void addLine(const Eigen::MatrixXd& LS) override;
	virtual int addDynamicLine(const Eigen::MatrixXd& LS) override;
	virtual void removeDynamicLine(int token) override;

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
