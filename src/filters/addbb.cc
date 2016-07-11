#include <igl/readOBJ.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char* argv[])
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	double margin = 0.1;
	int opt;

	while ((opt = getopt(argc, argv, "m:")) != -1) {
		switch (opt) {
			case 'm':
				margin = atof(optarg);
				break;
			case '?':
				fprintf(stderr, "Unrecognized option\n");
				return -1;
		};
	}
	fprintf(stderr, "Margin = %f\n", margin);

	readOBJ(stdin, V, F);
	Eigen::VectorXd maxV = V.colwise().maxCoeff();
	Eigen::VectorXd minV = V.colwise().minCoeff();
	double coords[3][2] = { {minV(0) - margin, maxV(0) + margin},
				{minV(1) - margin, maxV(1) + margin},
				{minV(2) - margin, maxV(2) + margin} };

	size_t rows = V.rows();
	Eigen::MatrixXd BV(8, 3);
	Eigen::ArrayXXi BF(12, 3);

	size_t i = 0;
	for(int ix = 0; ix < 2; ix++) {
		for(int iy = 0; iy < 2; iy++) {
			for(int iz = 0; iz < 2; iz++) {
				BV(i, 0) = coords[0][ix];
				BV(i, 1) = coords[1][iy];
				BV(i, 2) = coords[2][iz];
				i++;
			}
		}
	}
	BF << 1, 5, 2,
	      5, 6, 2,
	      5, 7, 6,
	      6, 7, 8,
	      2, 6, 4,
	      4, 6, 8,
	      3, 2, 4,
	      2, 3, 1,
	      1, 3, 5,
	      5, 3, 7,
	      4, 8, 7,
	      3, 4, 7;
	std::cout << V.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << BV.format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","v ","","","\n"))
		  << (F.array()+1).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"))
		  << (BF + rows).format(Eigen::IOFormat(Eigen::FullPrecision,Eigen::DontAlignCols," ","\n","f ","","","\n"));

	return 0;
}
