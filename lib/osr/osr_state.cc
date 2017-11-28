#include "osr_state.h"

namespace osr {

osr::Transform translate_state_to_transform(const StateVector& state)
{
	StateQuat rot(state(3), state(4), state(5), state(6));
	StateTrans trans(state(0), state(1), state(2));
	osr::Transform tf;
	tf.setIdentity();
	tf.rotate(rot);
	tf.pretranslate(trans);
	return tf;
}

StateVector
interpolate(const StateVector& pkey,
            const StateVector& nkey,
            StateScalar tau)
{
	StateTrans p0(pkey(0), pkey(1), pkey(2));
	StateTrans p1(nkey(0), nkey(1), nkey(2));
	StateTrans pinterp = p0 * (1-tau) + p1 * tau;
	StateQuat Qfrom { pkey(3), pkey(4), pkey(5), pkey(6) };
	StateQuat Qto { nkey(3), nkey(4), nkey(5), nkey(6) };
	Qfrom.normalize();
	Qto.normalize();
	StateQuat Qinterp = Qfrom.slerp(tau, Qto);
	StateVector ret;
	ret << pinterp(0), pinterp(1), pinterp(2),
	       Qinterp.w(), Qinterp.x(), Qinterp.y(), Qinterp.z();
	return ret;
}

double distance(const StateVector& lhv, const StateVector& rhv)
{
	double trdist = (lhv.segment<3>(0) - rhv.segment<3>(0)).norm();
	double dot = lhv.segment<4>(3).dot(rhv.segment<4>(3));
	double rotdist = std::abs(std::acos(dot)); // |theta/2|
	return trdist + rotdist;
}

std::tuple<StateTrans, AngleAxisVector>
differential(const StateVector& from, const StateVector& to)
{
	StateTrans tr = to.segment<3>(0) - from.segment<3>(0);
	StateQuat rot_from(from(3), from(4), from(5), from(6));
	StateQuat rot_to(to(3), to(4), to(5), to(6));
	StateQuat rot_delta = rot_to * rot_from.inverse();
	Eigen::AngleAxis<StateScalar> aa(rot_delta);
	AngleAxisVector aav = aa.axis() * aa.angle();
	return std::make_tuple(tr, aav);
}

}
