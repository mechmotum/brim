from __future__ import annotations

import pytest
from brim.bicycle import (
    FlatGround,
    KnifeEdgeWheel,
    MasslessCranks,
    NonHolonomicTire,
    RigidFrontFrame,
    RigidRearFrame,
    StationaryBicycle,
    WhippleBicycle,
)
from brim.brim import (
    BicycleRider,
    HolonomicHandGrips,
    HolonomicPedals,
    SideLeanSeat,
)
from brim.rider import (
    FixedSacrum,
    PinElbowStickLeftArm,
    PinElbowStickRightArm,
    PlanarPelvis,
    PlanarTorso,
    Rider,
    SphericalLeftHip,
    SphericalLeftShoulder,
    SphericalRightHip,
    SphericalRightShoulder,
    TwoPinStickLeftLeg,
    TwoPinStickRightLeg,
)
from brim.utilities.testing import ignore_point_warnings


class TestCompleteBicycleRider:
    @pytest.fixture()
    def _whipple_setup(self) -> None:
        self.bicycle = WhippleBicycle("bicycle")
        self.bicycle.front_frame = RigidFrontFrame("front_frame")
        self.bicycle.rear_frame = RigidRearFrame("rear_frame")
        self.bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
        self.bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
        self.bicycle.front_tire = NonHolonomicTire("front_tire")
        self.bicycle.rear_tire = NonHolonomicTire("rear_tire")
        self.bicycle.cranks = MasslessCranks("pedals")
        self.bicycle.ground = FlatGround("ground")

    @pytest.fixture()
    def _stationary_setup(self) -> None:
        self.bicycle = StationaryBicycle("bicycle")
        self.bicycle.front_frame = RigidFrontFrame("front_frame")
        self.bicycle.rear_frame = RigidRearFrame("rear_frame")
        self.bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
        self.bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
        self.bicycle.cranks = MasslessCranks("pedals")

    @pytest.fixture()
    def _rider_setup(self) -> None:
        self.rider = Rider("rider")
        self.rider.pelvis = PlanarPelvis("pelvis")
        self.rider.torso = PlanarTorso("torso")
        self.rider.left_arm = PinElbowStickLeftArm("left_arm")
        self.rider.right_arm = PinElbowStickRightArm("right_arm")
        self.rider.left_leg = TwoPinStickLeftLeg("left_leg")
        self.rider.right_leg = TwoPinStickRightLeg("right_leg")
        self.rider.sacrum = FixedSacrum("sacrum")
        self.rider.left_hip = SphericalLeftHip("left_hip")
        self.rider.right_hip = SphericalRightHip("right_hip")
        self.rider.left_shoulder = SphericalLeftShoulder("left_shoulder")
        self.rider.right_shoulder = SphericalRightShoulder("right_shoulder")

    @pytest.fixture()
    def _whipple_rider_setup(self, _whipple_setup, _rider_setup) -> None:
        self.br = BicycleRider("bicycle_rider")
        self.br.bicycle = self.bicycle
        self.br.rider = self.rider
        self.br.seat = SideLeanSeat("seat")
        self.br.pedals = HolonomicPedals("pedals")
        self.br.hand_grips = HolonomicHandGrips("steer_conn")

    @pytest.fixture()
    def _stationary_rider_setup(self, _stationary_setup, _rider_setup) -> None:
        self.br = BicycleRider("bicycle_rider")
        self.br.bicycle = self.bicycle
        self.br.rider = self.rider
        self.br.seat = SideLeanSeat("seat")
        self.br.pedals = HolonomicPedals("pedals")
        self.br.hand_grips = HolonomicHandGrips("steer_conn")

    def test_stationary_setup(self, _stationary_rider_setup) -> None:
        assert self.br.bicycle == self.bicycle
        assert self.br.rider == self.rider
        assert self.br.seat is not None
        assert self.br.pedals is not None
        assert self.br.hand_grips is not None
        self.br.define_all()

    def test_whipple_setup(self, _whipple_rider_setup) -> None:
        assert self.br.bicycle == self.bicycle
        assert self.br.rider == self.rider
        assert self.br.seat is not None
        assert self.br.pedals is not None
        assert self.br.hand_grips is not None
        self.br.define_all()

    def test_no_connections(self, _whipple_setup, _rider_setup) -> None:
        br = BicycleRider("bicycle_rider")
        br.bicycle = self.bicycle
        br.rider = self.rider
        br.define_all()

    def test_form_eoms(self, _whipple_rider_setup) -> None:
        self.br.define_all()
        system = self.br.to_system()
        system.q_ind = [*self.bicycle.q[:4], *self.bicycle.q[5:],
                        self.rider.left_hip.q[0], self.rider.left_leg.q[0],
                        self.rider.right_hip.q[0], self.rider.right_leg.q[0],
                        *self.rider.left_arm.q, *self.rider.right_arm.q,
                        *self.br.seat.q]
        system.q_dep = [self.bicycle.q[4],
                        *self.rider.left_hip.q[1:], self.rider.left_leg.q[1],
                        *self.rider.right_hip.q[1:], self.rider.right_leg.q[1],
                        *self.rider.left_shoulder.q, *self.rider.right_shoulder.q]
        system.u_ind = [self.bicycle.u[3], *self.bicycle.u[5:7],
                        self.rider.left_hip.u[0], self.rider.left_leg.u[0],
                        self.rider.right_hip.u[0], self.rider.right_leg.u[0],
                        *self.rider.left_arm.u, *self.rider.right_arm.u,
                        *self.br.seat.u]
        system.u_dep = [*self.bicycle.u[:3], self.bicycle.u[4], self.bicycle.u[7],
                        *self.rider.left_hip.u[1:], self.rider.left_leg.u[1],
                        *self.rider.right_hip.u[1:], self.rider.right_leg.u[1],
                        *self.rider.left_shoulder.u, *self.rider.right_shoulder.u]
        system.validate_system()
        with ignore_point_warnings():
            system.form_eoms()
