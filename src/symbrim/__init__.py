"""SymBRiM.

A Modular and Extensible Open-Source Framework for Creating Symbolic Bicycle-Rider
Models.
"""

__all__ = [
    "WhippleBicycle", "StationaryBicycle",

    "FlatGround",

    "RigidRearFrame",

    "RigidFrontFrame", "SuspensionRigidFrontFrame",

    "KnifeEdgeWheel", "ToroidalWheel",

    "InContactTire", "NonHolonomicTire",

    "MasslessCranks",

    "Rider",

    "PlanarPelvis",

    "PlanarTorso",

    "PinElbowStickLeftArm", "PinElbowStickRightArm",

    "TwoPinStickLeftLeg", "TwoPinStickRightLeg",

    "PinLeftHip", "PinRightHip", "SphericalLeftHip", "SphericalRightHip",

    "FlexAddLeftShoulder", "FlexAddRightShoulder",
    "FlexRotLeftShoulder", "FlexRotRightShoulder",
    "SphericalLeftShoulder", "SphericalRightShoulder",

    "FixedSacrum",

    "BicycleRider",

    "FixedSeat", "SideLeanSeat",

    "HolonomicHandGrips",

    "HolonomicPedals",
]

from symbrim.bicycle import (
    FlatGround,
    InContactTire,
    KnifeEdgeWheel,
    MasslessCranks,
    NonHolonomicTire,
    RigidFrontFrame,
    RigidRearFrame,
    StationaryBicycle,
    SuspensionRigidFrontFrame,
    ToroidalWheel,
    WhippleBicycle,
)
from symbrim.brim import (
    BicycleRider,
    FixedSeat,
    HolonomicHandGrips,
    HolonomicPedals,
    SideLeanSeat,
)
from symbrim.rider import (
    FixedSacrum,
    FlexAddLeftShoulder,
    FlexAddRightShoulder,
    FlexRotLeftShoulder,
    FlexRotRightShoulder,
    PinElbowStickLeftArm,
    PinElbowStickRightArm,
    PinLeftHip,
    PinRightHip,
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

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
