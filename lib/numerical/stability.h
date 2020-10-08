#ifndef NUMERICAL_STABILITY_H
#define NUMERICAL_STABILITY_H

inline bool fpclose(double f0, double f1)
{
	if (std::abs(f0 - f1) < 1e-6)
		return true;
	return false;
}

#endif
