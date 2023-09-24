import numpy as np
from faster_whisper import WhisperModel


def detect_language(model: WhisperModel, audio_data: np.ndarray) -> list[tuple[str, float]]:
    features = model.feature_extractor(audio_data)

    if not model.model.is_multilingual:
        return [("en", 1.0)]
    else:
        segment = features[:, : model.feature_extractor.nb_max_frames]
        encoder_output = model.encode(segment)
        results = model.model.detect_language(encoder_output)[0]
        return [(token[2:-2], prob) for (token, prob) in results]
