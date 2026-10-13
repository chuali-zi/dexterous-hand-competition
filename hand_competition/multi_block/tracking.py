"""Locally assigned, observation-only identities and conservative witnesses."""
from dataclasses import dataclass
from ..closed_loop.geometry import finite,distance,vector
from .perception import BlockObservation,COLORS


@dataclass(frozen=True)
class TrackView:
    track_id:str
    color:str
    center_m:tuple
    pixels:tuple
    camera_ids:tuple
    captured_s:float
    consecutive_frames:int
    unchanged_since_s:float
    robot_clear:bool
    @property
    def unchanged_for_s(self):return self.captured_s-self.unchanged_since_s


class Tracker:
    """IDs express observed continuity, never physical identity authentication.

    Any missing/ambiguous color retires its affected identity. Reappearance gets
    a new local ID, with no inherited transport/score/stability provenance.
    """
    def __init__(self,radius_m=.020,max_gap_s=.25,clearance_m=.080):
        if any(not finite(v) or v<=0 for v in (radius_m,max_gap_s,clearance_m)):raise ValueError('tracking budgets')
        self.radius_m=radius_m;self.max_gap_s=max_gap_s;self.clearance_m=clearance_m
        self.previous=();self.last_s=-1.;self.next_id=1
    def update(self,blocks,captured_s,tip):
        if not finite(captured_s) or captured_s<0 or captured_s<=self.last_s:raise ValueError('tracking clock reset/repeat')
        tip=vector(tip,3,'measured tip')
        if any(not isinstance(b,BlockObservation) for b in blocks):raise ValueError('visual observations required')
        blocks=tuple(sorted(blocks,key=lambda b:(COLORS.index(b.color),b.center_m,b.pixels)))
        if len(blocks)>10:raise ValueError('tracking object cap')
        gap=self.last_s>=0 and captured_s-self.last_s>self.max_gap_s+1e-9
        old=() if gap else self.previous
        current=[];issues=[]
        for color in COLORS:
            observations=[b for b in blocks if b.color==color]
            prior=[t for t in old if t.color==color]
            links={i:[j for j,t in enumerate(prior) if distance(b.center_m,t.center_m)<=self.radius_m]
                   for i,b in enumerate(observations)}
            if any(len(v)>1 for v in links.values()) or any(sum(j in v for v in links.values())>1 for j in range(len(prior))):
                issues.append(color+':temporal_ambiguity');continue
            for i,b in enumerate(observations):
                match=prior[links[i][0]] if links[i] else None
                clear=distance(b.center_m,tip)>=self.clearance_m
                same=(match is not None and match.pixels==b.pixels and match.camera_ids==b.camera_ids and match.robot_clear and clear and match.center_m==b.center_m)
                if match is None:
                    identity=f'local-{self.next_id:04}';self.next_id+=1
                else:identity=match.track_id
                current.append(TrackView(identity,color,b.center_m,b.pixels,b.camera_ids,captured_s,
                  match.consecutive_frames+1 if match else 1,match.unchanged_since_s if same else captured_s,clear))
        retired=tuple(t.track_id for t in self.previous if t.track_id not in {n.track_id for n in current})
        self.previous=tuple(current);self.last_s=captured_s
        return self.previous,tuple(issues),retired
