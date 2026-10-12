"""Local, reproducible sensor/action traces. Hashes are not signed attestations."""
import hashlib
import json
from pathlib import Path


def decode_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError(f'duplicate JSON key: {key}')
            value[key] = item
        return value
    def invalid(value): raise ValueError(f'non-finite JSON constant: {value}')
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def load_json(path):
    return decode_json(Path(path).read_text(encoding='utf-8'))


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
        self.version = manifest.get('schema_version', 1)
        manifest_path = self.root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        self.manifest_hash = digest(manifest_path.read_bytes())
        if self.version >= 2:
            (self.root/'config.json').write_text(json.dumps(manifest['config'], indent=2, allow_nan=False)+'\n')
        self._stream = (self.root/'events.jsonl').open('w', encoding='utf-8')
        self._previous = '0'*64; self.count = 0; self.observations = 0

    def event(self, kind, time_s, **payload):
        event = {'index': self.count, 'kind': kind, 'time_s': time_s,
                 'payload': payload, 'previous_sha256': self._previous}
        if self.version >= 2: event['manifest_sha256'] = self.manifest_hash
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
        robot = {'frame_id': 'base_link', 'tip_position_m': observation.tip}
        version = 1
        if observation.robot_captured_s is not None:
            version = 2
            robot.update(producer=observation.robot_producer, captured_s=observation.robot_captured_s,
                         sequence=observation.robot_sequence)
        self.event('observation', observation.time_s, producer=self.producer,
                   observation_schema_version=version, sensor_time_s=observation.sensor_time_s,
                   calibration_id=observation.calibration_id, frames=frames, robot=robot,
                   visual_position_m=estimate.position_m if estimate else None,
                   stereo_residual_m=estimate.stereo_residual_m if estimate else None,
                   vision_error=error)
        self.observations += 1

    @property
    def producer(self):
        return self._producer

    @producer.setter
    def producer(self, value): self._producer = value

    def finish(self, result, time_s=None):
        try:
            self.event('result', result['elapsed_s'] if time_s is None else time_s, result=result)
            (self.root/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        finally:
            self._stream.close()
