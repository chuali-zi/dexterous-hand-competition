"""Local, reproducible sensor/action traces. Hashes are not signed attestations."""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(data): return hashlib.sha256(data).hexdigest()


def write_ppm(path, width, height, rgb):
    Path(path).write_bytes(f'P6\n{width} {height}\n255\n'.encode('ascii')+rgb)


def read_ppm(path):
    data = Path(path).read_bytes(); parts = data.split(b'\n', 3)
    if len(parts) != 4 or parts[0] != b'P6' or parts[2] != b'255': raise ValueError('unsupported PPM')
    width, height = map(int, parts[1].split()); rgb = parts[3]
    if width <= 0 or height <= 0 or len(rgb) != width*height*3: raise ValueError('invalid PPM pixels')
    return width, height, rgb


class Trace:
    def __init__(self, directory, manifest):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root/'frames').mkdir()
        (self.root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        self._stream = (self.root/'events.jsonl').open('w', encoding='utf-8')
        self._previous = '0'*64; self.count = 0; self.observations = 0

    def event(self, kind, time_s, **payload):
        event = {'index': self.count, 'kind': kind, 'time_s': time_s,
                 'payload': payload, 'previous_sha256': self._previous}
        event['sha256'] = digest(canonical(event).encode('utf-8'))
        self._stream.write(canonical(event)+'\n'); self._stream.flush()
        self._previous = event['sha256']; self.count += 1

    def observation(self, observation, estimate=None, error=None):
        frames = []
        for frame in observation.frames:
            name = f'frames/{self.observations:05}-{frame.camera_id}.ppm'
            write_ppm(self.root/name, frame.width, frame.height, frame.rgb)
            frames.append({'camera_id': frame.camera_id, 'frame_id': frame.frame_id,
                           'sequence': frame.sequence, 'captured_s': frame.captured_s,
                           'width': frame.width, 'height': frame.height, 'encoding': 'rgb8',
                           'file': name, 'file_sha256': digest((self.root/name).read_bytes())})
        self.event('observation', observation.time_s, producer=self.producer,
                   calibration_id=observation.calibration_id, frames=frames,
                   robot={'frame_id': 'base_link', 'tip_position_m': observation.tip},
                   visual_position_m=estimate.position_m if estimate else None,
                   stereo_residual_m=estimate.stereo_residual_m if estimate else None,
                   vision_error=error)
        self.observations += 1

    @property
    def producer(self):
        return self._producer

    @producer.setter
    def producer(self, value): self._producer = value

    def finish(self, result):
        self.event('result', result['elapsed_s'], result=result)
        self._stream.close()
        (self.root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
