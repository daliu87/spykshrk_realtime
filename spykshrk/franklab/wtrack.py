import enum


@enum.unique
class WtrackArm(enum.Enum):
    center = 1
    left = 2
    right = 3


@enum.unique
class Direction(enum.Enum):
    outbound = 1
    inbound = 2


@enum.unique
class Rotation(enum.Enum):
    cw = 1
    ccw = 2


@enum.unique
class Order(enum.Enum):
    prev = 1
    main = 2
    next = 3