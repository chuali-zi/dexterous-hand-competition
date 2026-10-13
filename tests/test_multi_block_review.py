"""Reviewer regressions: independently rasterized RGB, no author fixtures.

Only Camera's calibration container is reused. Projection, rendering, hidden
scene choices and expectations below do not call production projection/planning.
"""
import unittest
from hand_competition.closed_loop.geometry import Camera
from hand_competition.multi_block.session import PlanningSession
from hand_competition.multi_block.planning import PlanConfig
from hand_competition.multi_block.proxy_geometry import rectangle_union_area,support_geometry,volume_intersects
from hand_competition.multi_block.tracking import TrackView

COLORS={'red':(230,30,30),'yellow':(230,200,30),'blue':(30,60,230)}


def calibration(half_baseline=.1):
    return tuple(Camera(name,(x,0.,1.),((1,0,0),(0,-1,0),(0,0,-1)),
                        fx=390.,fy=390.,cx=256.,cy=192.,width=512,height=384)
                 for name,x in (('left',-half_baseline),('right',half_baseline)))


def reviewer_packet(scene,sequence,cameras,visibility=None,crop_left=None):
    """scene is evaluator-only; per-view crops/visibility never enter packets."""
    stamp=round(sequence*.1,6);frames=[]
    for index,camera in enumerate(cameras):
        rgb=bytearray((35,35,35))*512*384
        for j,(color,(x,y,z)) in enumerate(scene):
            if visibility is not None and not visibility[index][j]:continue
            u=round(256+390*(x-camera.origin[0])/(1-z))
            v=round(192-390*y/(1-z))
            cut=0 if crop_left is None else crop_left[index][j]
            for py in range(max(0,v-5),min(384,v+6)):
                for px in range(max(0,u-5+cut),min(512,u+6)):
                    offset=3*(py*512+px);rgb[offset:offset+3]=bytes(COLORS[color])
        frames.append({'camera_id':camera.name,'frame_id':camera.name+'_optical','sequence':sequence,
                       'captured_s':stamp,'width':512,'height':384,'encoding':'rgb8','rgb':bytes(rgb)})
    return {'schema_version':2,'producer':'reviewer_rgb','calibration_id':'synthetic-pinhole-v1','time_s':stamp,
            'frames':frames,'robot':{'frame_id':'base_link','tip_position_m':(0.,.2,.3),
                    'producer':'reviewer_rgb','captured_s':stamp,'sequence':sequence}}


def observe_32(scene,cameras=None,**kwargs):
    cameras=cameras or calibration();session=PlanningSession(cameras,'reviewer_rgb');history=[]
    for sequence in range(1,33):
        raw=reviewer_packet(scene,sequence,cameras,**kwargs)
        _,snapshot=session.observe(raw,raw['time_s']);history.append(snapshot)
    return history


class ReviewerRGBRegressions(unittest.TestCase):
    def test_mb1_unmatched_blue_is_not_free_target_space(self):
        scene=[('red',(-.2,-.12,.025)),('blue',(.08,0.,.15))]
        last=observe_32(scene,visibility=((True,True),(True,False)))[-1]
        self.assertIn('blue:view_count_mismatch',last['perception_issues'])
        self.assertIsNone(last['decision']['candidate'])

    def test_mb2_actual_support_union_is_only_fifty_percent(self):
        scene=[('red',(.0725,0.,.025)),('red',(.1475,0.,.025)),('yellow',(-.2,-.12,.025))]
        # Independent 1D footprint lengths, y overlap is complete.
        self.assertAlmostEqual((.0975-.085+.135-.1225)/.05,.5)
        self.assertIsNone(observe_32(scene)[-1]['decision']['candidate'])

    def test_mb2_elevated_supports_penetrate_final_target_volume(self):
        scene=[('red',(.08,0.,.0325)),('red',(.14,0.,.0325)),('yellow',(-.2,-.12,.025))]
        last=observe_32(scene,calibration(.16))[-1]
        self.assertTrue(any(t['color']=='red' and t['center_m'][2]>.032 for t in last['tracks']))
        self.assertIsNone(last['decision']['candidate'])

    def test_mb3_asymmetric_clipping_cannot_certify_source_height(self):
        scene=[('red',(-.2,-.12,.06))]
        last=observe_32(scene,crop_left=((2,),(8,)))[-1]
        self.assertIsNone(last['decision']['candidate'])

    def test_complete_source_and_nominal_supports_remain_positive(self):
        source=observe_32([('red',(-.2,-.12,.025))])[-1]
        self.assertEqual(source['decision']['candidate']['target_slot_id'],'layer-0-slot-0')
        supported=observe_32([('red',(.08,0.,.025)),('red',(.14,0.,.025)),('yellow',(-.2,-.12,.025))])[-1]
        self.assertEqual(supported['decision']['candidate']['target_slot_id'],'layer-1-slot-0')

    def test_symmetric_clipping_cannot_pass_equal_view_areas(self):
        last=observe_32([('red',(-.2,-.12,.025))],crop_left=((2,),(2,)))[-1]
        self.assertIsNone(last['decision']['candidate'])
        self.assertIn('red:component_incomplete',last['perception_issues'])
        self.assertTrue(last['spatial_uncertainties'])

    def test_tiny_component_is_retained_as_uncertain_obstacle(self):
        cameras=calibration();session=PlanningSession(cameras,'reviewer_rgb')
        raw=reviewer_packet([('red',(-.2,-.12,.025))],1,cameras)
        rgb=bytearray(raw['frames'][0]['rgb']);rgb[3*(192*512+328):3*(192*512+328)+3]=bytes(COLORS['blue'])
        raw['frames'][0]['rgb']=bytes(rgb)
        _,snapshot=session.observe(raw,.1)
        self.assertIn('blue:view_count_mismatch',snapshot['perception_issues'])
        self.assertEqual(snapshot['spatial_uncertainties'][0]['component_evidence'][0]['area'],1)
        self.assertIsNone(snapshot['decision']['candidate'])

    def test_identical_rgb_does_not_authenticate_same_physical_object(self):
        cameras=calibration()
        ordinary=[('red',(-.2,-.125,.025))]
        # Two different heights/points, each visible in just its corresponding
        # view. Independent inverse scalar equations place their 11px markers
        # at exactly the normal source's pixels. No stereo system can identify
        # which hidden scene produced byte-for-byte identical RGB.
        different=[('red',(-.1+(216-256)*.94/390,-(242-192)*.94/390,.06)),
                   ('red',(.1+(136-256)*.82/390,-(242-192)*.82/390,.18))]
        normal=reviewer_packet(ordinary,1,cameras)
        false_pair=reviewer_packet(different,1,cameras,visibility=((True,False),(False,True)))
        self.assertEqual([f['rgb'] for f in normal['frames']],[f['rgb'] for f in false_pair['frames']])
        result=observe_32(different,cameras,visibility=((True,False),(False,True)))[-1]['decision']
        self.assertIsNotNone(result['candidate'])  # conditional geometry only
        self.assertIs(result['correspondence_authenticity_verified'],False)
        self.assertIs(result['physical_component_completeness_verified'],False)
        self.assertIs(result['execution_available'],False)


class ActualProxyGeometryTests(unittest.TestCase):
    def track(self,identity,center):
        return TrackView(identity,'red',center,((1,1),(2,2)),('left','right'),3.2,32,.1,True)

    def test_overlap_union_is_not_sum_and_y_offset_matters(self):
        self.assertAlmostEqual(rectangle_union_area([(0,1,0,1),(.5,1.5,0,1)]),1.5)
        parents=[self.track('a',(.08,.008,.025)),self.track('b',(.14,.008,.025))]
        check=support_geometry((.11,0,.075),parents,PlanConfig())
        self.assertAlmostEqual(check['support_fraction'],.672)
        self.assertFalse(check['valid'])

    def test_top_plane_errors_and_positive_volume_intersections_are_not_exempt(self):
        parents=[self.track('a',(.08,0,.0325)),self.track('b',(.14,0,.0325))]
        check=support_geometry((.11,0,.075),parents,PlanConfig())
        self.assertEqual(check['reason'],'support_contact_height_mismatch')
        self.assertAlmostEqual(check['top_plane_errors_m'][0],.0075)
        self.assertTrue(volume_intersects((.11,0,.075),parents[0].center_m,(.05,.05,.05)))
        self.assertFalse(volume_intersects((.11,0,.075),(.08,0,.025),(.05,.05,.05)))
