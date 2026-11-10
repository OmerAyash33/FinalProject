"""
Audio cue for kicks. The chest-cam records sound, and a kick is a sharp broadband click (ball hitting the boot).

Why it exists: on this footage a real shot leaves almost no ball track (the ball is blurred, the camera swings with the body,
the ball is seen again only tiny a second later), so the speed rule in event_detection cannot fire. The click is still there.
The click alone is NOT specific (every hold end has some sharp noise: steps, touches, shouts; measured on the user's labels,
loudness at the kick separates shots from passes only at AUC ~0.5-0.65), so event_detection combines it with
"the goal is in view for a while after the kick". Everything tunable is in event_params.py section 4b.

Stage 1.5: extract_audio_features(video, npz) is run once per video (main.py / tools/extract_data_only.py), the result is a small
`<video>_audio.npz` next to the raw CSV, so Stage 2 never needs the video file. AudioTrack reads it.
"""
import os

import numpy as np

import event_params as P

AUDIO_SUFFIX = "_audio.npz"
HOP_S = 0.01          # feature hop (10 ms); the files store one value per hop
_WIN_S = 0.02
_Z_WINDOW = 401       # 4 s of hops: local median / MAD used for the robust z-score


def audio_path_for(raw_csv):
    """`<dir>/<video>_data.csv` -> `<dir>/<video>_audio.npz`."""
    return raw_csv[:-len("_data.csv")] + AUDIO_SUFFIX if raw_csv.endswith("_data.csv") else raw_csv + AUDIO_SUFFIX


def _decode_mono(video_path):
    import av  # imported lazily: the pipeline still runs (without the audio cue) if PyAV is not installed
    container = av.open(video_path)
    try:
        if not container.streams.audio:
            return None, None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=stream.rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            return None, None
        return np.concatenate(chunks).astype(np.float32), int(stream.rate)
    finally:
        container.close()


def compute_features(x, sr):
    """10 ms-hop features of a mono signal: `z` = robust z-score of the spectral flux (100 Hz-8 kHz): how many local MADs a moment's
    onset energy sits above the 4 s median around it (a click is 20-150, ordinary noise is < 10); `low` = 80-500 Hz level in dB
    relative to the whole video's median. Returns dict(z, low) as float32 arrays, index i = time i * HOP_S."""
    from scipy import signal
    from scipy.ndimage import median_filter
    hop, win = int(sr * HOP_S), int(sr * _WIN_S)
    f, _, S = signal.spectrogram(x, sr, nperseg=win * 2, noverlap=win * 2 - hop, scaling="spectrum")
    band = (f >= 100) & (f < 8000)
    L = 10 * np.log10(S[band] + 1e-12)
    flux = np.concatenate([[0.0], np.maximum(L[:, 1:] - L[:, :-1], 0).mean(0)])
    med = median_filter(flux, size=_Z_WINDOW, mode="nearest")
    mad = median_filter(np.abs(flux - med), size=_Z_WINDOW, mode="nearest") + 1e-3
    low_band = (f >= 80) & (f < 500)
    low = 10 * np.log10(S[low_band].sum(0) + 1e-12)
    return {"z": ((flux - med) / mad).astype(np.float32), "low": (low - np.median(low)).astype(np.float32)}


def extract_audio_features(video_path, out_npz):
    """Decode the video's audio track and store its kick features. Returns True if written (False: no audio / PyAV missing)."""
    try:
        x, sr = _decode_mono(video_path)
    except ImportError:
        return False
    if x is None:
        return False
    np.savez_compressed(out_npz, **compute_features(x, sr), hop_s=np.float32(HOP_S))
    return True


class AudioTrack:
    """Per-video kick cue read from `<video>_audio.npz`. `load` returns None when there is no file (cue disabled for that video)."""

    def __init__(self, z, low, hop_s):
        self.z, self.low, self.hop_s = z, low, float(hop_s)

    @classmethod
    def load(cls, npz_path):
        if not os.path.exists(npz_path):
            return None
        d = np.load(npz_path)
        return cls(d["z"], d["low"], float(d["hop_s"]))

    def kick_z(self, frame, fps, pre_s=None, post_s=None):
        """Strongest click (robust z) in [frame - pre_s, frame + post_s]; 0 if the window lies outside the recording."""
        pre = P.AUDIO_KICK_PRE_S if pre_s is None else pre_s
        post = P.AUDIO_KICK_POST_S if post_s is None else post_s
        t0 = frame / fps
        i0, i1 = max(0, int((t0 - pre) / self.hop_s)), min(len(self.z), int((t0 + post) / self.hop_s) + 1)
        return float(self.z[i0:i1].max()) if i1 > i0 else 0.0
