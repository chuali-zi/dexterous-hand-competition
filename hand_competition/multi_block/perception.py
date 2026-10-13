"""RGB component stereo association; neither object IDs nor poses are inputs."""
from dataclasses import dataclass
from itertools import permutations
from ..closed_loop.geometry import Camera,triangulate,finite,vector,distance
from ..closed_loop.vision import components

COLORS=('red','yellow','blue','green')


@dataclass(frozen=True)
class VisionConfig:
    workspace:tuple=(-.4,.4,-.25,.25,.01,.40)
    max_residual_m:float=.003
    min_area_px:int=20
    max_objects:int=10
    max_same_color:int=4
    def __post_init__(self):
        object.__setattr__(self,'workspace',vector(self.workspace,6,'vision workspace'))
        if any(self.workspace[i]>=self.workspace[i+1] for i in (0,2,4)):raise ValueError('vision workspace ordering')
        if not finite(self.max_residual_m) or self.max_residual_m<=0:raise ValueError('vision residual')
        if type(self.min_area_px)is not int or self.min_area_px<1:raise ValueError('component area')
        if type(self.max_objects)is not int or not 1<=self.max_objects<=10:raise ValueError('object cap')
        if type(self.max_same_color)is not int or not 1<=self.max_same_color<=4:raise ValueError('same-color permutation cap')


@dataclass(frozen=True)
class BlockObservation:
    color:str
    center_m:tuple
    pixels:tuple
    camera_ids:tuple
    residual_m:float
    def __post_init__(self):
        if self.color not in COLORS:raise ValueError('unknown color')
        object.__setattr__(self,'center_m',vector(self.center_m,3,'visual center'))
        if len(self.pixels)!=2 or len(self.camera_ids)!=2 or len(set(self.camera_ids))!=2:raise ValueError('two distinct views required')
        object.__setattr__(self,'pixels',tuple(vector(p,2,'pixel') for p in self.pixels))
        if not finite(self.residual_m) or self.residual_m<0:raise ValueError('visual residual')


@dataclass(frozen=True)
class PerceptionResult:
    blocks:tuple
    issues:tuple


def associate(detections,cameras,config):
    """Require one unique full bijection per color; never pick a best-looking tie.

    Partial visibility blocks that color, not unrelated colors. Workspace and
    epipolar consistency are engineering hypotheses, not object truth.
    """
    if len(cameras)!=2 or len(detections)!=2:raise ValueError('two-view association only')
    if any(len(view)>config.max_objects for view in detections):
        return PerceptionResult((),('object_count_exceeds_cap',))
    if any(d.color not in COLORS for view in detections for d in view):raise ValueError('invalid detection color')
    found=[];issues=[]
    for color in COLORS:
        a=sorted((d for d in detections[0] if d.color==color),key=lambda d:d.centroid)
        b=sorted((d for d in detections[1] if d.color==color),key=lambda d:d.centroid)
        if len(a)!=len(b):issues.append(color+':view_count_mismatch');continue
        if not a:continue
        if len(a)>config.max_same_color:issues.append(color+':same_color_cap');continue
        edges={}
        for i,first in enumerate(a):
            for j,second in enumerate(b):
                try:point,residual=triangulate(cameras[0],first.centroid,cameras[1],second.centroid,config.max_residual_m)
                except ValueError:continue
                if all(config.workspace[2*k]<=point[k]<=config.workspace[2*k+1] for k in range(3)):
                    edges[i,j]=(point,residual)
        valid=[]
        for assignment in permutations(range(len(b))):
            if all((i,j)in edges for i,j in enumerate(assignment)):
                valid.append(assignment)
                if len(valid)>1:break
        if len(valid)!=1:
            issues.append(color+(':stereo_ambiguous' if valid else ':no_geometric_bijection'));continue
        for i,j in enumerate(valid[0]):
            point,residual=edges[i,j]
            found.append(BlockObservation(color,point,(a[i].centroid,b[j].centroid),
                         tuple(c.name for c in cameras),residual))
    return PerceptionResult(tuple(sorted(found,key=lambda x:(COLORS.index(x.color),x.center_m,x.pixels))),tuple(issues))


def observe_many(observation,cameras,config=VisionConfig()):
    by_name={frame.camera_id:frame for frame in observation.frames}
    if len(cameras)!=2 or set(by_name)!={c.name for c in cameras}:raise ValueError('two calibrated RGB views required')
    detected=tuple(components(by_name[c.name],config.min_area_px) for c in cameras)
    return associate(detected,cameras,config)
