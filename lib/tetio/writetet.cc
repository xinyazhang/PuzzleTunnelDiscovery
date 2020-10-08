/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "writetet.h"
#include <fstream>
#include <stdexcept>

using std::endl;
using std::runtime_error;

void
write_node(std::ostream& fout,
	   const Eigen::MatrixXd& V)
{
	fout << V.rows() << "  3  0  0" << endl;
	fout.precision(17);
	for(int i = 0; i < V.rows(); i++) {
		fout << "    " << i
		     << "    " << V(i, 0)
		     << "    " << V(i, 1)
		     << "    " << V(i, 2)
		     << endl;
	}
	fout << "# pickup by writetet" << endl;
}

void
write_edge(std::ostream& fout,
	   const Eigen::MatrixXi& E)
{
	// FIXME: Set boundary markers correctly
	fout << E.rows() << "  1" << endl;
	for(int i = 0; i < E.rows(); i++) {
		fout << "    " << i
		     << "    " << E(i,0)
		     << "    " << E(i,1)
		     << "   1"
		     << endl;
	}
}

void
write_ele(std::ostream& fout,
	  const Eigen::MatrixXi& P)
{
	fout << P.rows() << "  4  0" << endl;
	for(int i = 0; i < P.rows(); i++) {
		fout << "    " << i
		     << "    " << P(i,0)
		     << "    " << P(i,1)
		     << "    " << P(i,2)
		     << "    " << P(i,3)
		     << endl;
	}
}

void
writetet(const std::string& oprefix,
	 const Eigen::MatrixXd& V,
	 const Eigen::MatrixXi& E,
	 const Eigen::MatrixXi& P)
{
	{
		std::ofstream fout(oprefix+".node");
		if (!fout.is_open())
			throw runtime_error("Cannot open "+oprefix+".node to write");
		write_node(fout, V);
	}
	{
		std::ofstream fout(oprefix+".edge");
		if (!fout.is_open())
			throw runtime_error("Cannot open "+oprefix+".edge to write");
		write_edge(fout, E);
	}
	{
		std::ofstream fout(oprefix+".ele");
		if (!fout.is_open())
			throw runtime_error("Cannot open "+oprefix+".ele to write");
		write_ele(fout, P);
	}
}

void
write_face(std::ostream& fout,
	   const Eigen::MatrixXi& F,
	   const Eigen::VectorXi* FBM)
{
	fout << F.rows() << "  1" << endl;
	for(int i = 0; i < F.rows(); i++) {
		fout << "    " << i
		     << "    " << F(i,0)
		     << "    " << F(i,1)
		     << "    " << F(i,2);
		if (FBM)
			fout << "   " << (*FBM)(i);
		else
			fout << "   1";
		fout << endl;
	}
}

void
writetet_face(const std::string& oprefix,
	      const Eigen::MatrixXi& F,
	      const Eigen::VectorXi* FBM)
{
	std::ofstream fout(oprefix+".face");
	if (!fout.is_open())
		throw runtime_error("Cannot open "+oprefix+".face to write");
	write_face(fout, F, FBM);
}
