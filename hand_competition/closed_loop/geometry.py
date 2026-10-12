"""Small pinhole/stereo geometry implementation, independent of robot SDKs."""
from dataclasses import dataclass
import math


def add(a, b): return tuple(x + y for x, y in zip(a, b))
def sub(a, b): return tuple(x - y for x, y in zip(a, b))
def mul(a, k): return tuple(x * k for x in a)
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def distance(a, b): return norm(sub(a, b))
def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def unit(a):
    length = norm(a)
    if length < 1e-12: raise ValueError('zero vector')
    return mul(a, 1 / length)


@dataclass(frozen=True)
class Camera:
    name: str
    origin: tuple
    rotation: tuple  # base -> optical, rows x-right / y-down / z-forward
    fx: float = 180.0
    fy: float = 180.0
    cx: float = 80.0
    cy: float = 60.0
    width: int = 160
    height: int = 120
    calibration_id: str = 'synthetic-pinhole-v1'

    def __post_init__(self):
        values = (*self.origin, *(v for row in self.rotation for v in row),
                  self.fx, self.fy, self.cx, self.cy)
        if len(self.origin) != 3 or len(self.rotation) != 3 or any(len(r) != 3 for r in self.rotation):
            raise ValueError('invalid camera transform dimensions')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError('non-finite camera calibration')
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError('invalid camera intrinsics')
        for i, row in enumerate(self.rotation):
            for j, other in enumerate(self.rotation):
                if abs(dot(row, other) - int(i == j)) > 1e-6:
                    raise ValueError('camera rotation is not orthonormal')
        if distance(cross(self.rotation[0], self.rotation[1]), self.rotation[2]) > 1e-6:
            raise ValueError('camera rotation is not right handed')

    def project(self, point):
        relative = sub(point, self.origin)
        x, y, z = (dot(row, relative) for row in self.rotation)
        if z <= 0: raise ValueError('point behind camera')
        return (self.fx*x/z + self.cx, self.fy*y/z + self.cy)

    def ray(self, uv):
        optical = ((uv[0]-self.cx)/self.fx, (uv[1]-self.cy)/self.fy, 1.0)
        base = tuple(sum(self.rotation[j][i]*optical[j] for j in range(3)) for i in range(3))
        return self.origin, unit(base)


def look_at(name, origin, target):
    forward = unit(sub(target, origin))
    right = unit(cross(forward, (0, 0, 1)))
    down = cross(forward, right)
    return Camera(name, origin, (right, down, forward))


def triangulate(first, uv1, second, uv2, max_residual_m=0.02):
    o1, d1 = first.ray(uv1); o2, d2 = second.ray(uv2)
    w = sub(o1, o2); b = dot(d1, d2); denominator = 1-b*b
    if denominator < 0.0025: raise ValueError('stereo rays almost parallel')
    d = dot(d1, w); e = dot(d2, w)
    t1 = (b*e-d)/denominator; t2 = (e-b*d)/denominator
    if t1 <= 0 or t2 <= 0: raise ValueError('triangulated point behind camera')
    p1 = add(o1, mul(d1, t1)); p2 = add(o2, mul(d2, t2))
    residual = distance(p1, p2)
    if residual > max_residual_m: raise ValueError('stereo correspondence residual too large')
    return mul(add(p1, p2), 0.5), residual
