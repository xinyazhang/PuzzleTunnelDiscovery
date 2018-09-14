#include <fcl/fcl.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

using Scalar = float;
using Vector3 = fcl::Vector3<Scalar>;
using BVType = fcl::OBBRSS<Scalar>;
using Model = fcl::BVHModel<BVType>;
using Transform = Eigen::Transform<Scalar, 3, Eigen::AffineCompact>;
using VMatrix = Eigen::Matrix<Scalar, -1, 3, Eigen::RowMajor>;

using std::cout;
using std::endl;

void sancheck_1()
{
	// Create two vertices
	Model htri, vtri;
	std::vector<Vector3> htv { { 0.0, 0.0, 0.0}, { 1.0, 0.0, 0.0},  { 0.0, 0.0, 1.0} };
	std::vector<Vector3> vtv { { 0.0, 0.0, 0.0}, { 1.0, 0.0, 0.0},  { 0.0, 1.0, 0.0} };
	std::vector<fcl::Triangle> tri_indices;
	tri_indices.resize(1);
	tri_indices[0].set(0, 1, 2);

	htri.beginModel();
	htri.addSubModel(htv, tri_indices);
	htri.endModel();

	vtri.beginModel();
	vtri.addSubModel(vtv, tri_indices);
	vtri.endModel();

	fcl::CollisionRequest<Scalar> req;
	fcl::CollisionResult<Scalar> res;
	req.enable_contact = true;
	req.num_max_contacts = 4;

	Transform htf;
	Transform vtf;

	htf.setIdentity();
	vtf.setIdentity();

	Eigen::Vector3f vpos;
	vpos << -0.49, -0.5, 0.5;
	vtf.translate(vpos);

	// size_t ret = fcl::collide(&htri, htf, &vtri, vtf, req, res);
	size_t ret = fcl::collide(&vtri, vtf, &htri, htf, req, res);

	cout << "Collision (expect true) " << (ret > 0) << endl;
	cout << "Number of contacts (expect 2) " << res.numContacts() << endl;
	std::vector<fcl::Contact<Scalar>> cts;
	res.getContacts(cts);
	for (size_t i = 0; i < cts.size(); i++) {
		const auto& ct = cts[i];
		cout << "Contact " << i << " : " << ct.pos[0] << " " << ct.pos[1] << " " << ct.pos[2];
		cout << " CN " << ct.normal[0] << " " << ct.normal[1] << " " << ct.normal[2];
		cout << " PD " << ct.penetration_depth;
		cout << endl;
	}
}

void sancheck_2();

int main()
{
	sancheck_1();
	sancheck_2();
}

extern
int tri_tri_intersect_with_isectline(float V0[3],float V1[3],float V2[3],
				     float U0[3],float U1[3],float U2[3],int *coplanar,
				     float isectpt1[3],float isectpt2[3]);

int tri_tri_intersect_cc(Eigen::Vector3f v0, Eigen::Vector3f v1, Eigen::Vector3f v2,
                          Eigen::Vector3f u0, Eigen::Vector3f u1, Eigen::Vector3f u2,
                          int *coplanar,
                          Eigen::Vector3f& isectpt1_out, Eigen::Vector3f& isectpt2_out)
{
	float isectpt1[3];
	float isectpt2[3];

	int ret = tri_tri_intersect_with_isectline(v0.data(), v1.data(), v2.data(),
	                                           u0.data(), u1.data(), u2.data(),
	                                           coplanar,
	                                           isectpt1, isectpt2);

	isectpt1_out << isectpt1[0], isectpt1[1], isectpt1[2];
	isectpt2_out << isectpt2[0], isectpt2[1], isectpt2[2];

	return ret;
}

void sancheck_2()
{
	Transform htf;
	Transform vtf;

	htf.setIdentity();
	vtf.setIdentity();

	Eigen::Vector3f vpos;
	vpos << -0.49, -0.5, 0.5;
	vtf.translate(vpos);

	VMatrix template_htv, template_vtv;
	template_htv.resize(3,3);
	template_vtv.resize(3,3);
	template_htv << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
	template_vtv << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;

	VMatrix htv = (htf * template_htv.transpose()).transpose();
	VMatrix vtv = (vtf * template_vtv.transpose()).transpose();

	int cp;
	Eigen::Vector3f isectpt1;
	Eigen::Vector3f isectpt2;

	tri_tri_intersect_cc(htv.row(0), htv.row(1), htv.row(2),
	                     vtv.row(0), vtv.row(1), vtv.row(2),
	                     &cp, isectpt1, isectpt2);
	cout << isectpt1.transpose() << "  " << isectpt2.transpose() << endl;
}
