"""Independent closed-form RGB fixtures; hidden labels are evaluator-only."""
import random
from hand_competition.closed_loop.geometry import Camera

RGB={'red':(230,30,30),'yellow':(230,200,30),'blue':(30,60,230),'green':(30,200,60)}
ROTATION=((1,0,0),(0,-1,0),(0,0,-1))
def calibrated_cameras(half_baseline=.10,focal=390.):
    return tuple(Camera(name,(x,0,1.),ROTATION,fx=focal,fy=focal,cx=256.,cy=192.,width=512,height=384)
                 for name,x in (('camera_left',-half_baseline),('camera_right',half_baseline)))

CAMERAS=calibrated_cameras()

def initial_scene(seed=0):
    colors=['red']*4+['yellow']*3+['blue']*2+['green']
    positions=[(x,y,.025) for x in (-.28,-.20,-.12) for y in (-.15,-.05,.05,.15)]
    rng=random.Random(seed);rng.shuffle(positions)
    return [{'hidden_label':f'evaluator-only-{i}','color':c,'center':positions[i]} for i,c in enumerate(colors)]


def pixels_for(point,camera_index,cameras=CAMERAS):
    """Analytic equations; does not call production Camera.project/ray."""
    x,y,z=point;camera=cameras[camera_index];origin_x=camera.origin[0]
    return (camera.cx+camera.fx*(x-origin_x)/(1-z),camera.cy-camera.fy*y/(1-z))


def packet(scene,sequence,time_s,hidden_colors=(),pixel_jitter=0,seed=0,cameras=CAMERAS):
    frames=[];rng=random.Random(seed+sequence*271)
    for camera_index,camera in enumerate(cameras):
        data=bytearray((35,35,35))*camera.width*camera.height
        for item in scene:
            if camera_index==1 and item['color'] in hidden_colors:continue
            u,v=pixels_for(item['center'],camera_index,cameras);u=round(u)+rng.randint(-pixel_jitter,pixel_jitter);v=round(v)+rng.randint(-pixel_jitter,pixel_jitter)
            for y in range(max(0,v-5),min(camera.height,v+6)):
                for x in range(max(0,u-5),min(camera.width,u+6)):
                    index=3*(y*camera.width+x);data[index:index+3]=bytes(RGB[item['color']])
        frames.append({'camera_id':camera.name,'frame_id':camera.name+'_optical','sequence':sequence,'captured_s':time_s,
          'width':camera.width,'height':camera.height,'encoding':'rgb8','rgb':bytes(data)})
    return {'schema_version':2,'producer':'offline_multi_rgb_fixture_v1','calibration_id':'synthetic-pinhole-v1','time_s':time_s,
            'frames':frames,'robot':{'frame_id':'base_link','tip_position_m':(0.,.20,.30),'producer':'offline_multi_rgb_fixture_v1',
                                    'captured_s':time_s,'sequence':sequence}}


def goal_scene(layer_count=4):
    """Independently written 4-3-2-1 coordinates, not production blueprint."""
    scene=[]
    for layer,(count,color) in enumerate(((4,'red'),(3,'yellow'),(2,'blue'),(1,'green'))):
        if layer>=layer_count:break
        for i in range(count):scene.append({'hidden_label':f'goal-eval-{layer}-{i}','color':color,
                   'center':(.08+.06*(i+.5*layer),0.,.025+.05*layer)})
    return scene


def staged_scene(layers=0,seed=0):
    target=goal_scene(layers);source=initial_scene(seed)
    used={c:sum(item['color']==c for item in target) for c in RGB}
    remaining=[]
    for item in source:
        if used[item['color']]:used[item['color']]-=1
        else:remaining.append(item)
    return target+remaining
