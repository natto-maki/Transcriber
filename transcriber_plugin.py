import numpy as np


FLAG_AUDIO = 0x1
FLAG_VAD = 0x2
FLAG_SPEECH_SEGMENT = 0x4

FLAG_ADD_TAB = 0x10


class Plugin:
    """
    note: callbacks (functions beginning with on_*) are invoked via the thread pool.
    """
    def __init__(self, **kwargs):
        self._sampling_rate = kwargs["__sampling_rate"]
        self._ui_language = kwargs["__ui_language"]
        self._data_dir = kwargs["__data_dir"]
        self._input_language = kwargs["__input_language"]
        self._output_language = kwargs["__output_language"]

    def injection_point(self) -> int:
        """
        This function should return the logical OR of the FLAG_* constants
        at the point where this plugin triggers the hook.
        :return: The logical OR of the FLAG_* constants
        """
        return 0

    def on_audio_frame(self, device_index: int, timestamp: float, audio_data: np.ndarray):
        """
        The function receives audio data acquired from the input device as it is.

        required FLAG_AUDIO

        :param device_index: Index starting from 0 to identify the input device when multiple inputs are used
        :param timestamp: Timestamp of audio_data, based on time.time()
        :param audio_data: Raw audio data; dtype = float32, shape = (frame_size,);
            frame_size will be always 512 in the current implementation and the sampling rate will be 16000.
            The sampling rate can also be obtained from self._sampling_rate.
        """
        pass

    def on_vad_frame(self, timestamp: float, audio_data: np.ndarray | None):
        """
        This function receives the speech data of a sequence of utterances detected by the VAD.

        If audio_data is None, it means keep-alive frames,
        which are generated approximately every second,
        indicating that no valid speech data was observed during that time.
        Since frames representing valid speech in the VAD are generated at the end of a speech segment,
        it is not possible to determine whether a certain amount of time has elapsed without silence
        or whether it is in the middle of a long speech segment.
        By checking keep-alive frames instead of timeouts, it is possible to distinguish between each event.

        required FLAG_VAD

        :param timestamp: Timestamp of audio_data, based on time.time()
        :param audio_data: Raw audio data or None for keep-alive; dtype = float32, shape = (*,)
        """
        pass

    def on_speech_segment(self, tm0: float, tm1: float, person_name: str | None, text: str | None):
        """
        This function receives the transcribed text;
        the utterances contained in the speech detected by the VAD are separated by speaker,
        and this callback is called for each sentence of appropriate length.

        Keep-alive frames can be identified by tm0.
        Since the transcription process is time-consuming,
        it is not recommended to use a real-time timer to measure time
        (e.g., to measure the silence time between conversations).
        If such a measurement is necessary, it can be done using the tm1 timestamp of the keep-alive frame.

        required FLAG_SPEECH_SEGMENT

        :param tm0: Start time of this segment, or -1.0 for keep-alive
        :param tm1: End time of this segment
        :param person_name: The name of the estimated speaker based on the database at the time of detection,
            where None indicates that no speaker was stored in the database that could be considered as a match.
            If not enough speaker feature vectors were stored in the database,
            person_name may report incorrect values,
            and the speaker name corresponding to this text may be rewritten later
            when additional clustering is performed
            (currently there is no way to retrieve that rewritten speaker name via the plugin interfaces).
        :param text: Transcribed text or None for keep-alive
        """
        pass

    def tab_name(self) -> str:
        """
        Returns the name given to gradio.Tab() if the plugin provides a user interface.

        required FLAG_ADD_TAB

        :return: Tab name
        """
        return ""

    def build_tab(self):
        """
        Constructs the user interface for this plugin.
        This function is called under 'with gradio.Tab()'
        and allows you to place any component in the tabs provided exclusively for this plugin.

        required FLAG_ADD_TAB
        """
        pass

    def read_state(self, read_parameters: str) -> str:
        """
        Reads the status of this plugin.
        Used for debugging purposes and to retrieve the processing results by this plugin
        when the application is running in server mode.

        :return: Any text
        """
        return ""
