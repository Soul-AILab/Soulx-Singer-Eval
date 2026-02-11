import parselmouth
import numpy as np
from scipy.interpolate import interp1d


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_f0_features_using_parselmouth(audio, cfg, speed=1):
    """Using parselmouth to extract the f0 feature.
    Args:
        audio
        mel_len
        hop_length
        fs
        f0_min
        f0_max
        speed(default=1)
    Returns:
        f0: numpy array of shape (frame_len,)
        pitch_coarse: numpy array of shape (frame_len,)
    """
    hop_size = int(np.round(cfg.hop_size * speed))

    # Calculate the time step for pitch extraction
    time_step = hop_size / cfg.sample_rate * 1000

    f0 = (
        parselmouth.Sound(audio, cfg.sample_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=cfg.f0_min,
            pitch_ceiling=cfg.f0_max,
        )
        .selected_array["frequency"]
    )
    return f0


def get_cents(f0_hz):
    """
    F_{cent} = 1200 * log2 (F/440)

    Reference:
        APSIPA'17, Perceptual Evaluation of Singing Quality
    """
    voiced_f0 = f0_hz[f0_hz != 0]
    return 1200 * np.log2(voiced_f0 / 440)


def get_pitch_sub_median(f0_hz):
    """
    f0_hz: (,T)
    """
    f0_cent = get_cents(f0_hz)
    return f0_cent - np.median(f0_cent)


def same_t_in_true_and_est(func):
    def new_func(true_t, true_f, est_t, est_f):
        assert type(true_t) is np.ndarray
        assert type(true_f) is np.ndarray
        assert type(est_t) is np.ndarray
        assert type(est_f) is np.ndarray

        if len(true_t) == len(est_t):
            time_diff = np.abs(true_t - est_t)
            if np.max(time_diff) < 1e-4:
                return func(true_t, true_f, est_t, est_f)
            
        interpolated_f = interp1d(
            est_t, est_f, bounds_error=False, kind='nearest', fill_value=0
        )(true_t)
        return func(true_t, true_f, true_t, interpolated_f)
    return new_func


def gross_pitch_error_frames(true_t, true_f, est_t, est_f, eps=1e-8):
    voiced_frames = true_voiced_frames(true_t, true_f, est_t, est_f)
    true_f_p_eps = [x + eps for x in true_f]
    pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
    return voiced_frames & pitch_error_frames


def voicing_decision_error_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) != (true_f != 0)


def true_voiced_frames(true_t, true_f, est_t, est_f):
    return (est_f != 0) & (true_f != 0)


def add_basic_stats(dic):
    mean = np.mean(list(dic.values()))
    std = np.std(list(dic.values()))
    max = np.max(list(dic.values()))
    argmax = float(np.argmax(list(dic.values())))
    min = np.min(list(dic.values()))
    argmin = float(np.argmin(list(dic.values())))
    dic['stats'] = {'mean':mean, 'std':std, 'max':max, 'argmax':argmax, 'min':min, 'argmin':argmin}
    return dic
