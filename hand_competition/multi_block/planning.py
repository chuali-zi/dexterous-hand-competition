"""Color/dependency planning from current visual tracks; never a score judge."""
from dataclasses import dataclass,asdict,fields
from ..closed_loop.geometry import vector,finite,distance
from .perception import COLORS


@dataclass(frozen=True)
class PlanConfig:
    block_size_m:tuple=(.05,.05,.05)
    workspace:tuple=(-.40,.40,-.25,.25,.0,.40)
    supply_xy:tuple=(-.35,-.03,-.20,.20)
    build_xy:tuple=(.03,.34,-.20,.20)
    base_anchor_m:tuple=(.08,0.,.025)
    layer_counts:tuple=(4,3,2,1)
    layer_colors:tuple=('red','yellow','blue','green')
    pitch_x_m:float=.06
    clearance_m:float=.005
    target_xy_tolerance_m:float=.008
    target_z_tolerance_m:float=.008
    source_height_tolerance_m:float=.008
    observed_unchanged_s:float=3.
    association_radius_m:float=.020
    max_observation_gap_s:float=.25
    max_observation_age_s:float=.25
    max_sensor_skew_s:float=.03
    hand_clearance_m:float=.08
    session_budget_s:float=300.
    min_support_fraction:float=.70
    official_scene_confirmed:bool=False
    material_dimensions_source:str='illustrative_fixture_assumption_not_official'
    def __post_init__(self):
        for name,n in (('block_size_m',3),('workspace',6),('supply_xy',4),('build_xy',4),('base_anchor_m',3)):
            object.__setattr__(self,name,vector(getattr(self,name),n,name))
        for name in ('layer_counts','layer_colors'):
            if not isinstance(getattr(self,name),(tuple,list)):raise ValueError('layer sequence')
            object.__setattr__(self,name,tuple(getattr(self,name)))
        excluded={'block_size_m','workspace','supply_xy','build_xy','base_anchor_m','layer_counts','layer_colors','official_scene_confirmed','material_dimensions_source'}
        for f in fields(self):
            if f.name not in excluded and (not finite(getattr(self,f.name)) or getattr(self,f.name)<=0):raise ValueError('invalid '+f.name)
        if any(v<=0 for v in self.block_size_m):raise ValueError('block dimensions')
        if any(self.workspace[i]>=self.workspace[i+1] for i in (0,2,4)):raise ValueError('workspace ordering')
        if len(self.layer_counts)!=len(self.layer_colors) or not self.layer_counts or len(self.layer_counts)>4:raise ValueError('layer specification')
        if any(type(n)is not int or n<1 for n in self.layer_counts) or sum(self.layer_counts)>10:raise ValueError('ten-block planning cap')
        if any(self.layer_counts[i]!=self.layer_counts[i-1]-1 for i in range(1,len(self.layer_counts))):raise ValueError('supported layout is descending pyramid')
        if any(c not in COLORS for c in self.layer_colors):raise ValueError('layer colors')
        if type(self.official_scene_confirmed)is not bool or self.official_scene_confirmed:
            raise ValueError('this provisional offline blueprint cannot claim official scene confirmation')
        if not isinstance(self.material_dimensions_source,str) or not self.material_dimensions_source.strip():raise ValueError('dimension provenance')
        for zone in (self.supply_xy,self.build_xy):
            if not self.workspace[0]<=zone[0]<zone[1]<=self.workspace[1] or not self.workspace[2]<=zone[2]<zone[3]<=self.workspace[3]:raise ValueError('zone ordering/workspace')
        if not (self.supply_xy[1]<=self.build_xy[0] or self.build_xy[1]<=self.supply_xy[0]
                or self.supply_xy[3]<=self.build_xy[2] or self.build_xy[3]<=self.supply_xy[2]):raise ValueError('supply/build regions overlap')
        if self.pitch_x_m<self.block_size_m[0] or self.pitch_x_m>=2*self.block_size_m[0]:raise ValueError('pitch creates collision or unsupported upper row')
        support_fraction=(2*self.block_size_m[0]-self.pitch_x_m)/self.block_size_m[0]
        if not 0<self.min_support_fraction<=1 or support_fraction+1e-9<self.min_support_fraction:raise ValueError('illustrative support footprint insufficient')
        if self.target_xy_tolerance_m>=self.pitch_x_m/2 or self.target_z_tolerance_m>=self.block_size_m[2]/2:raise ValueError('slot tolerances mask neighboring slots')
        if self.observed_unchanged_s<3 or self.max_sensor_skew_s>self.max_observation_age_s or self.max_observation_gap_s>self.max_observation_age_s:raise ValueError('observation timing')
        if abs(self.base_anchor_m[2]-self.block_size_m[2]/2)>1e-9:raise ValueError('fixture table is base z=0')
        for slot in build_slots_unchecked(self):
            if not contained(slot.center_m,self.block_size_m,self.build_xy,self.clearance_m):raise ValueError('target footprint outside build region')
            if any(not self.workspace[2*i]<=slot.center_m[i]-self.block_size_m[i]/2 or not slot.center_m[i]+self.block_size_m[i]/2<=self.workspace[2*i+1] for i in range(3)):raise ValueError('target volume outside workspace')
    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value)-{f.name for f in fields(cls)}:raise ValueError('unknown plan config field')
        return cls(**value)


@dataclass(frozen=True)
class Slot:
    slot_id:str
    layer:int
    index:int
    color:str
    center_m:tuple
    supports:tuple


def build_slots_unchecked(config):
    slots=[]
    for layer,(count,color) in enumerate(zip(config.layer_counts,config.layer_colors)):
        for i in range(count):
            slots.append(Slot(f'layer-{layer}-slot-{i}',layer,i,color,
                (config.base_anchor_m[0]+config.pitch_x_m*(i+layer/2),config.base_anchor_m[1],config.base_anchor_m[2]+layer*config.block_size_m[2]),
                () if layer==0 else (f'layer-{layer-1}-slot-{i}',f'layer-{layer-1}-slot-{i+1}')))
    return tuple(slots)


def build_slots(config):return build_slots_unchecked(config)


def contained(center,size,zone,margin):
    return (zone[0]+margin<=center[0]-size[0]/2 and center[0]+size[0]/2<=zone[1]-margin
            and zone[2]+margin<=center[1]-size[1]/2 and center[1]+size[1]/2<=zone[3]-margin)


def xy_intersects(a,b,size,margin):
    return abs(a[0]-b[0])<size[0]+2*margin and abs(a[1]-b[1])<size[1]+2*margin


def near_slot(track,slot,config):
    return ((track.center_m[0]-slot.center_m[0])**2+(track.center_m[1]-slot.center_m[1])**2)**.5<=config.target_xy_tolerance_m and abs(track.center_m[2]-slot.center_m[2])<=config.target_z_tolerance_m


def witnessed(track,config):
    return track.robot_clear and track.unchanged_for_s>=config.observed_unchanged_s-1e-9


def plan(tracks,config=PlanConfig()):
    """Pure replan from the current snapshot; no claimed executed placement.

    Geometry may suggest support, but legal robot transport/contact/official
    scoring cannot be established here. A candidate is not an SDK command.
    """
    slots=build_slots(config);occupancy={};supported=set();blocked={};used=set()
    for slot in slots:
        possible=[t for t in tracks if near_slot(t,slot,config)]
        if len(possible)>1:blocked[slot.slot_id]='ambiguous_occupancy';continue
        if not possible:continue
        t=possible[0]
        if t.track_id in used:raise ValueError('one track cannot occupy multiple slots')
        used.add(t.track_id);occupancy[slot.slot_id]=t.track_id
        if t.color!=slot.color:blocked[slot.slot_id]='wrong_color_occupied'
        elif not witnessed(t,config):blocked[slot.slot_id]='unverified_observation_window'
        elif any(parent not in supported for parent in slot.supports):blocked[slot.slot_id]='support_dependency_missing'
        else:supported.add(slot.slot_id)
    decisions=[]
    for slot in slots:
        if slot.slot_id in occupancy or slot.slot_id in blocked:continue
        if any(parent not in supported for parent in slot.supports):
            blocked[slot.slot_id]='support_dependency_missing';continue
        support_ids={occupancy[parent] for parent in slot.supports}
        # Vertical descent column can be obstructed even with no final-volume
        # overlap; never repair a lower row through surviving upper blocks.
        obstacles=[t for t in tracks if t.track_id not in support_ids
                   and xy_intersects(t.center_m,slot.center_m,config.block_size_m,config.clearance_m)
                   and t.center_m[2]+config.block_size_m[2]/2>slot.center_m[2]-config.block_size_m[2]/2+1e-6]
        if obstacles:blocked[slot.slot_id]='occupied_target_or_descent_column';continue
        sources=[]
        for t in tracks:
            if t.color!=slot.color or not witnessed(t,config):continue
            if not contained(t.center_m,config.block_size_m,config.supply_xy,config.clearance_m):continue
            if abs(t.center_m[2]-config.base_anchor_m[2])>config.source_height_tolerance_m:continue
            if any(other.track_id!=t.track_id and xy_intersects(t.center_m,other.center_m,config.block_size_m,config.clearance_m)
                   and abs(other.center_m[2]-t.center_m[2])<config.block_size_m[2] for other in tracks):continue
            sources.append(t)
        if not sources:blocked[slot.slot_id]='no_current_confirmed_clear_source';continue
        source=min(sources,key=lambda t:(distance(t.center_m,slot.center_m),t.center_m,t.track_id))
        decisions.append((slot.layer,slot.index,slot,source))
    candidate=None
    if decisions:
        _,_,slot,source=min(decisions,key=lambda row:row[:2])
        candidate={'source_track_id':source.track_id,'source_color':source.color,'source_center_m':source.center_m,
                   'target_slot_id':slot.slot_id,'target_center_m':slot.center_m,'required_support_slots':slot.supports,
                   'observation_captured_s':source.captured_s}
    return {'kind':'candidate' if candidate else 'geometric_goal_observed' if len(supported)==len(slots) else 'wait_or_blocked',
            'candidate':candidate,'geometric_occupancy':occupancy,'geometrically_supported_slots':sorted(supported),
            'remaining_slots':[s.slot_id for s in slots if s.slot_id not in supported],'blocked':blocked,
            'official_score':None,'robot_actions_executed':0,'execution_available':False,'physical_support_verified':False,
            'robot_collision_verified':False,'collision_scope':'axis_aligned_block_proxy_and_vertical_descent_column_only',
            'transport_provenance_verified':False}
