from __future__ import annotations

import pytest
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


class TestCompleteRider:
    @pytest.fixture()
    def _setup(self) -> None:
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

    def test_setup(self, _setup) -> None:
        assert self.rider.pelvis is not None
        assert self.rider.torso is not None
        assert self.rider.left_arm is not None
        assert self.rider.right_arm is not None
        assert self.rider.left_leg is not None
        assert self.rider.right_leg is not None
        assert self.rider.sacrum is not None
        assert self.rider.left_hip is not None
        assert self.rider.right_hip is not None
        assert self.rider.left_shoulder is not None
        assert self.rider.right_shoulder is not None
        self.rider.define_all()

    @pytest.mark.parametrize("define_required", [True, False])
    def test_minimal(self, define_required) -> None:
        rider = Rider("rider")
        if define_required:
            rider.pelvis = PlanarPelvis("pelvis")
            rider.define_all()
        else:
            with pytest.raises((ValueError, AttributeError)):
                rider.define_all()

    def test_form_eoms(self, _setup) -> None:
        self.rider.define_all()
        system = self.rider.to_system()
        system.validate_system()
        system.form_eoms()
