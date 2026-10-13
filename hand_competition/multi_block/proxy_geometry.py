"""Observed axis-aligned block proxies, not physical/official contact judging."""
from ..closed_loop.geometry import vector

EPSILON=1e-9  # arithmetic equality only, not sensor/official geometry tolerance


def footprint(center,size):
    return (center[0]-size[0]/2,center[0]+size[0]/2,center[1]-size[1]/2,center[1]+size[1]/2)


def rectangle_union_area(rectangles):
    """Exact sweep union: overlapping supporting areas are never double counted."""
    rectangles=[r for r in rectangles if r[0]<r[1] and r[2]<r[3]]
    xs=sorted({x for r in rectangles for x in r[:2]});area=0.
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((r[2],r[3]) for r in rectangles if r[0]<right and r[1]>left)
        length=0.;end=None
        for low,high in intervals:
            if end is None or low>end:length+=high-low;end=high
            elif high>end:length+=high-end;end=high
        area+=(right-left)*length
    return area


def volume_intersects(first,second,size):
    first=vector(first,3,'first block center');second=vector(second,3,'second block center')
    return all(size[i]-abs(first[i]-second[i])>EPSILON for i in range(3))


def support_geometry(center,parents,config):
    """Actual parent top planes and footprint union, before any ID exemption.

    Zero-depth top contact is allowed within numerical epsilon. A sensed height
    mismatch is rejected, never repaired by waiting or a pose tolerance band.
    """
    size=config.block_size_m;target=footprint(center,size);bottom=center[2]-size[2]/2
    if not parents:
        valid=abs(bottom)<=EPSILON
        return {'valid':valid,'reason':None if valid else 'table_contact_height_mismatch',
                'support_fraction':1. if valid else 0.,'top_plane_errors_m':[]}
    errors=[p.center_m[2]+size[2]/2-bottom for p in parents]
    rectangles=[]
    for p in parents:
        parent=footprint(p.center_m,size)
        rectangles.append((max(target[0],parent[0]),min(target[1],parent[1]),
                           max(target[2],parent[2]),min(target[3],parent[3])))
    fraction=rectangle_union_area(rectangles)/(size[0]*size[1])
    if any(abs(error)>EPSILON for error in errors):reason='support_contact_height_mismatch'
    elif any(r[0]>=r[1] or r[2]>=r[3] for r in rectangles):reason='required_parent_has_no_contact_footprint'
    elif fraction+EPSILON<config.min_support_fraction:reason='actual_support_union_insufficient'
    else:reason=None
    return {'valid':reason is None,'reason':reason,'support_fraction':fraction,'top_plane_errors_m':errors}
