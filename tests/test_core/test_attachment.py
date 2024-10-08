import pytest
from sympy.physics.mechanics import (
    Dyadic,
    Inertia,
    PinJoint,
    Point,
    ReferenceFrame,
    Vector,
)

from symbrim.core.attachment import Attachment, Hub, MasslessBody


class TestMasslessBody:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.frame = ReferenceFrame("frame")
        self.mc = Point("masscenter")
        self.body = MasslessBody("body", self.mc, self.frame)

    def test_init(self) -> None:
        assert self.body.name == "body"
        assert self.body.frame == self.frame
        assert self.body.masscenter == self.mc
        assert self.body.mass == 0
        assert self.body.central_inertia == Dyadic(0)
        assert self.body.potential_energy == 0
        assert self.body.kinetic_energy(self.frame) == 0
        assert self.body.linear_momentum(self.frame) == Vector(0)
        assert self.body.angular_momentum(self.mc, self.frame) == Vector(0)

    def test_fixed_mass(self) -> None:
        with pytest.raises(AttributeError):
            self.body.mass = 1

    def test_fixed_inertia(self) -> None:
        inertia = Inertia.from_inertia_scalars(self.mc, self.frame, 1, 1, 1)
        with pytest.raises(AttributeError):
            self.body.central_inertia = inertia.dyadic
        with pytest.raises(AttributeError):
            self.body.inertia = inertia

    def test_fixed_potential_energy(self) -> None:
        with pytest.raises(AttributeError):
            self.body.potential_energy = 1


@pytest.mark.parametrize("cls, kwargs", [(Attachment, {}), (Hub, {"axis": "x"})])
class TestAttachment:
    @pytest.fixture
    def _setup(self, cls, kwargs) -> None:
        self.frame = ReferenceFrame("frame")
        self.point = Point("point")
        self.attachment = cls(self.frame, self.point, **kwargs)

    @pytest.mark.usefixtures("_setup")
    def test_init(self) -> None:
        assert self.attachment.frame == self.frame
        assert self.attachment.point == self.point
        assert self.attachment.point.vel(self.frame) == Vector(0)
        assert self.attachment[0] == self.frame
        assert self.attachment[1] == self.point

    def test_from_name(self, cls, kwargs) -> None:
        attachment = cls.from_name("attachment", **kwargs)
        assert attachment.frame.name == "attachment_frame"
        assert attachment.point.name == "attachment_point"

    @pytest.mark.parametrize("set_body_name", [True, False])
    @pytest.mark.usefixtures("_setup")
    def test_to_valid_joint_arg(self, set_body_name) -> None:
        kwargs = {"name": "my_body"} if set_body_name else {}
        parent = self.attachment.to_valid_joint_arg(**kwargs)
        child = MasslessBody("body", Point("point2"), ReferenceFrame("frame2"))
        PinJoint("joint", parent, child)
        assert parent.name == "my_body" if set_body_name else "massless_body"

    @pytest.mark.usefixtures("_setup")
    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            self.attachment.frame = ReferenceFrame("frame2")
        with pytest.raises(AttributeError):
            self.attachment.point = Point("point2")

    @pytest.mark.parametrize(("point", "frame"), [
        (Point("point1"), Point("point2")),
        (ReferenceFrame("frame1"), ReferenceFrame("frame2")),
    ])
    def test_invalid_init_args(self, cls, kwargs, point, frame):
        with pytest.raises(TypeError):
            cls(frame, point, **kwargs)

    def test_invalid_name(self, cls, kwargs) -> None:
        with pytest.raises(ValueError):
            cls.from_name(2.2, **kwargs)

class TestHub:

    @pytest.mark.parametrize(("axis", "direction", "times"), [
        ("+x", 0, 1),
        ("-x", 0, -1),
        ("+y", 1, 1),
        ("-y", 1, -1),
        ("+z", 2, 1),
        ("-z", 2, -1),
        ("x", 0, 1),
        ("y", 1, 1),
        ("z", 2, 1),
    ])
    def test_axis_as_string(self, axis: str, times: int, direction: int) -> None:
        hub = Hub(ReferenceFrame("frame"), Point("point"), axis)
        vectors = (hub.frame.x, hub.frame.y, hub.frame.z)
        assert hub.axis == times * vectors[direction]

    def test_axis_as_vector(self) -> None:
        frame = ReferenceFrame("frame")
        hub = Hub(frame, Point("point"), frame.x + frame.z)
        assert hub.axis == frame.x + frame.z

    @pytest.mark.parametrize("axis", ["a", 1])
    def test_invalid_axis(self, axis) -> None:
        with pytest.raises((TypeError, ValueError)):
            Hub(ReferenceFrame("frame"), Point("point"), axis)

    def test_from_name(self) -> None:
        hub = Hub.from_name("hub", "x")
        assert hub.frame.name == "hub_frame"
        assert hub.point.name == "hub_point"
        assert hub.axis == hub.frame.x
