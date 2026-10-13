"""RGB component stereo association; neither object IDs nor poses are inputs."""
from dataclasses import dataclass
from collections import deque
from itertools import permutations
from ..closed_loop.geometry import Camera,triangulate,finite,vector,distance
from ..closed_loop.vision import classify

COLORS=('red','yellow','blue','green')


@dataclass(frozen=True)
class VisionConfig:
    workspace:tuple=(-.4,.4,-.25,.25,.01,.40)
    max_residual_m:float=.003
    min_area_px:int=20
    max_objects:int=10
    max_same_color:int=4
    component_model:str='offline_full_rectangle_11px_v1'
    def __post_init__(self):
        object.__setattr__(self,'workspace',vector(self.workspace,6,'vision workspace'))
        if any(self.workspace[i]>=self.workspace[i+1] for i in (0,2,4)):raise ValueError('vision workspace ordering')
        if not finite(self.max_residual_m) or self.max_residual_m<=0:raise ValueError('vision residual')
        if type(self.min_area_px)is not int or self.min_area_px<1:raise ValueError('component area')
        if type(self.max_objects)is not int or not 1<=self.max_objects<=10:raise ValueError('object cap')
        if type(self.max_same_color)is not int or not 1<=self.max_same_color<=4:raise ValueError('same-color permutation cap')
        if self.component_model!='offline_full_rectangle_11px_v1':raise ValueError('unreviewed component completeness model')


@dataclass(frozen=True)
class Component:
    color:str
    centroid:tuple
    area:int
    bounds:tuple  # inclusive pixel x_min,x_max,y_min,y_max
    touches_border:bool=False


def color_components(frame):
    """Keep even tiny colored components: discarded pixels are not free space."""
    labels=[classify(*frame.rgb[i:i+3]) for i in range(0,len(frame.rgb),3)]
    seen=bytearray(len(labels));result=[];width=frame.width
    for index,color in enumerate(labels):
        if color is None or seen[index]:continue
        todo=deque([index]);seen[index]=1;area=sx=sy=0
        xmin=xmax=index%width;ymin=ymax=index//width
        while todo:
            p=todo.popleft();x=p%width;y=p//width
            area+=1;sx+=x;sy+=y;xmin=min(xmin,x);xmax=max(xmax,x);ymin=min(ymin,y);ymax=max(ymax,y)
            neighbors=[]
            if x:neighbors.append(p-1)
            if x+1<width:neighbors.append(p+1)
            if y:neighbors.append(p-width)
            if y+1<frame.height:neighbors.append(p+width)
            for q in neighbors:
                if not seen[q] and labels[q]==color:seen[q]=1;todo.append(q)
        result.append(Component(color,(sx/area,sy/area),area,(xmin,xmax,ymin,ymax),
                      xmin==0 or ymin==0 or xmax==width-1 or ymax==frame.height-1))
    return tuple(result)


def complete_component(item,config):
    # Exact, explicitly provisional marker silhouette. No claim that a real
    # cube must be 11x11 or that equal silhouettes authenticate correspondence.
    if not isinstance(item,Component) or type(item.area)is not int or item.area<config.min_area_px:return False
    if item.touches_border or len(item.bounds)!=4:return False
    xmin,xmax,ymin,ymax=item.bounds
    return (abs(xmax-xmin+1-11)<=1e-9 and abs(ymax-ymin+1-11)<=1e-9 and item.area==121
            and abs(item.centroid[0]-(xmin+xmax)/2)<=1e-9 and abs(item.centroid[1]-(ymin+ymax)/2)<=1e-9)


@dataclass(frozen=True)
class SpatialUncertainty:
    reason:str
    bounds_m:tuple
    component_evidence:tuple=()
    def __post_init__(self):
        object.__setattr__(self,'bounds_m',vector(self.bounds_m,6,'unknown obstacle bounds'))
        if any(self.bounds_m[i]>=self.bounds_m[i+1] for i in (0,2,4)):raise ValueError('unknown obstacle bounds ordering')
        if not isinstance(self.reason,str) or not self.reason:raise ValueError('unknown obstacle reason')


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
    uncertainties:tuple=()


def perception_result(blocks,issues,detections,cameras,config):
    unknown=[]
    for issue in issues:
        color=issue.split(':')[0]
        evidence=tuple({'camera_id':camera.name,'color':d.color,'centroid':d.centroid,'area':d.area,
                         'bounds':getattr(d,'bounds',None)}
                        for camera,view in zip(cameras,detections) for d in view if color not in COLORS or d.color==color)
        # No certified object silhouette/depth interval exists. Conservatively
        # retain the entire workspace instead of removing an unlocalized block.
        unknown.append(SpatialUncertainty(issue,config.workspace,evidence))
    return PerceptionResult(tuple(blocks),tuple(issues),tuple(unknown))


def associate(detections,cameras,config):
    """Require one unique full bijection per color; never pick a best-looking tie.

    Partial visibility can preserve unrelated tracks, but carries unresolved
    occupied-space evidence to the planner. It never certifies free space.
    """
    if len(cameras)!=2 or len(detections)!=2:raise ValueError('two-view association only')
    if any(len(view)>config.max_objects for view in detections):
        return perception_result((),('object_count_exceeds_cap',),detections,cameras,config)
    if any(d.color not in COLORS for view in detections for d in view):raise ValueError('invalid detection color')
    found=[];issues=[]
    for color in COLORS:
        a=sorted((d for d in detections[0] if d.color==color),key=lambda d:d.centroid)
        b=sorted((d for d in detections[1] if d.color==color),key=lambda d:d.centroid)
        if len(a)!=len(b):issues.append(color+':view_count_mismatch');continue
        if not a:continue
        if len(a)>config.max_same_color:issues.append(color+':same_color_cap');continue
        if any(not complete_component(d,config) for d in a+b):issues.append(color+':component_incomplete');continue
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
    return perception_result(sorted(found,key=lambda x:(COLORS.index(x.color),x.center_m,x.pixels)),issues,detections,cameras,config)


def observe_many(observation,cameras,config=VisionConfig()):
    by_name={frame.camera_id:frame for frame in observation.frames}
    if len(cameras)!=2 or set(by_name)!={c.name for c in cameras}:raise ValueError('two calibrated RGB views required')
    detected=tuple(color_components(by_name[c.name]) for c in cameras)
    return associate(detected,cameras,config)
