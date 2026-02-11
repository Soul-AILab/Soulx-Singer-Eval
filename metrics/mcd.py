from pymcd.mcd import Calculate_MCD


def extract_mcd(audio_ref, audio_deg):
    """Extract Mel-Cepstral Distance for a two given audio.
    Args:
        audio_ref: The given reference audio. It is an audio path.
        audio_deg: The given synthesized audio. It is an audio path.
    """

    mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    mcd_value = mcd_toolbox.calculate_mcd(audio_ref, audio_deg)
    return mcd_value
