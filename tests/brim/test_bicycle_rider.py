from __future__ import annotations

import pytest
from brim.bicycle import (
    FlatGround,
    KnifeEdgeWheel,
    NonHolonomicTyreModel,
    RigidFrontFrame,
    RigidRearFrame,
    SimplePedals,
    WhippleBicycle,
)
from brim.brim import (
    BicycleRider,
    HolonomicPedalsConnection,
    HolonomicSteerConnection,
    SideLeanConnection,
)
from brim.rider import (
    FixedPelvisToTorso,
    PinElbowStickLeftArm,
    PinElbowStickRightArm,
    Rider,
    SimpleRigidPelvis,
    SimpleRigidTorso,
    SphericalLeftHip,
    SphericalLeftShoulder,
    SphericalRightHip,
    SphericalRightShoulder,
    TwoPinStickLeftLeg,
    TwoPinStickRightLeg,
)
from brim.utilities import to_system


class TestCompleteBicycleRider:
    @pytest.fixture()
    def _bicycle_setup(self) -> None:
        self.bicycle = WhippleBicycle("bicycle")
        self.bicycle.front_frame = RigidFrontFrame("front_frame")
        self.bicycle.rear_frame = RigidRearFrame("rear_frame")
        self.bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
        self.bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
        self.bicycle.front_tyre = NonHolonomicTyreModel("front_tyre")
        self.bicycle.rear_tyre = NonHolonomicTyreModel("rear_tyre")
        self.bicycle.pedals = SimplePedals("pedals")
        self.bicycle.ground = FlatGround("ground")

    @pytest.fixture()
    def _rider_setup(self) -> None:
        self.rider = Rider("rider")
        self.rider.pelvis = SimpleRigidPelvis("pelvis")
        self.rider.torso = SimpleRigidTorso("torso")
        self.rider.left_arm = PinElbowStickLeftArm("left_arm")
        self.rider.right_arm = PinElbowStickRightArm("right_arm")
        self.rider.left_leg = TwoPinStickLeftLeg("left_leg")
        self.rider.right_leg = TwoPinStickRightLeg("right_leg")
        self.rider.pelvis_to_torso = FixedPelvisToTorso("pelvis_to_torso")
        self.rider.left_hip = SphericalLeftHip("left_hip")
        self.rider.right_hip = SphericalRightHip("right_hip")
        self.rider.left_shoulder = SphericalLeftShoulder("left_shoulder")
        self.rider.right_shoulder = SphericalRightShoulder("right_shoulder")

    @pytest.fixture()
    def _bicycle_rider_setup(self, _bicycle_setup, _rider_setup) -> None:
        self.br = BicycleRider("bicycle_rider")
        self.br.bicycle = self.bicycle
        self.br.rider = self.rider
        self.br.seat_connection = SideLeanConnection("seat_conn")
        self.br.pedal_connection = HolonomicPedalsConnection("pedals_conn")
        self.br.steer_connection = HolonomicSteerConnection("steer_conn")

    def test_setup(self, _bicycle_rider_setup) -> None:
        assert self.br.bicycle == self.bicycle
        assert self.br.rider == self.rider
        assert self.br.seat_connection is not None
        assert self.br.pedal_connection is not None
        assert self.br.steer_connection is not None
        self.br.define_all()

    def test_no_connections(self, _bicycle_setup, _rider_setup) -> None:
        br = BicycleRider("bicycle_rider")
        br.bicycle = self.bicycle
        br.rider = self.rider
        br.define_all()

    def test_form_eoms(self, _bicycle_rider_setup) -> None:
        self.br.define_all()
        system = to_system(self.br)
        system.q_ind = [*self.bicycle.q[:4], *self.bicycle.q[5:],
                        self.rider.left_hip.q[0], self.rider.left_leg.q[0],
                        self.rider.right_hip.q[0], self.rider.right_leg.q[0],
                        self.rider.left_arm.q, self.rider.right_arm.q,
                        self.br.seat_connection.q]
        system.q_dep = [self.bicycle.q[4],
                        *self.rider.left_hip.q[1:], self.rider.left_leg.q[1],
                        *self.rider.right_hip.q[1:], self.rider.right_leg.q[1],
                        *self.rider.left_shoulder.q, *self.rider.right_shoulder.q]
        system.u_ind = [self.bicycle.u[3], *self.bicycle.u[5:7],
                        self.rider.left_hip.u[0], self.rider.left_leg.u[0],
                        self.rider.right_hip.u[0], self.rider.right_leg.u[0],
                        self.rider.left_arm.u, self.rider.right_arm.u,
                        self.br.seat_connection.u]
        system.u_dep = [*self.bicycle.u[:3], self.bicycle.u[4], self.bicycle.u[7],
                        *self.rider.left_hip.u[1:], self.rider.left_leg.u[1],
                        *self.rider.right_hip.u[1:], self.rider.right_leg.u[1],
                        *self.rider.left_shoulder.u, *self.rider.right_shoulder.u]
        system.validate_system()
        system.form_eoms()