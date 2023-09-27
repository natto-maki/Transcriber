# Transcriber

Japanese README is [here](README.ja.md)

This is an application that automatically performs audio transcription and minutes generation.
The transcribed content is output in real time, 
and a summary is automatically generated for a certain number of sentences.
It can also automatically identify the speaker (experimental feature).

### What's new in v0.2.0

- Automatic input language detection has been implemented.<br/>
  Settings: \[experimental\] Enable language estimation of input speech
- Real-time translation using LLM is now available when input language is different from the output language.<br/>
  Settings: \[experimental\] Enable real-time translation when input and output languages are different
- Added playback of recorded audio for each sentence.
- Several problems related to startup stoppage have been fixed.

<img width="400" alt="screenshot0" src="https://github.com/natto-maki/Transcriber/assets/145577363/7284e9b3-2adb-46d7-b230-28fde2ebe238">
<img width="400" alt="screenshot1" src="https://github.com/natto-maki/Transcriber/assets/145577363/c800597b-11aa-4ce5-821b-c1bf0b4b9045">
<img width="400" alt="screenshot2" src="https://github.com/natto-maki/Transcriber/assets/145577363/0dff60f4-a2f2-42d1-ad8c-9bb331eb2c0f">


## System overview

- For the VAD that detects human voices, it is using [Silero VAD](https://github.com/snakers4/silero-vad).
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) is used for transcription. 
  In order to process in real time and to avoid having backorders, 
  it is recommended to use a GPU, but CPU execution is also possible with the lower-precision model. 
  And, although it is a bit tricky to set up,
  it is also possible to run the process on GPU machines on the local network via gRPC communication.
- For calculating feature vectors for speaker identification, 
  you can choose either [SpeechBrain](https://github.com/speechbrain/speechbrain) or
  [Pyannote.audio](https://github.com/pyannote/pyannote-audio/). 
  The clustering algorithm for speaker identification is originally written with DBSCAN as the backend.
- To generate a summary, it uses the [OpenAI API](https://platform.openai.com/docs/introduction).
  An OpenAI API Key is required for this.
- The UI is written in [Gradio](https://www.gradio.app/) and is accessed from browsers to a local server.

## System requirement

- Python 3.11 or later
- GPU VRAM<=8GB (optional)
- OpenAI API Key; It will work without API Key, but in that case only the transcription will work.
- Microphone (required) and speakers (optional). 
  Note that the microphone and speakers will be used on the machine running the application, 
  not those via the browser.

## Installation

### Setup of virtual environment (optional)

Please set up `venv` or something if you need to avoid contaminating the environment

In the case of launching a docker container or virtual machine, 
make sure that port 7860 is visible, which the local server is listening on.
You may also need to mount `/dev/snd` on the host to make the microphone visible.<br/>
(After installing the dependent packages, try running `test_sound_device.py` and check whether it passes)

### Installing dependent packages

Go to the top directory of the repository and execute the following command:

```commandline
pip3 install -r requirements.txt
```

After that, run `test_sound_device.py` to test the microphone input.
If `pass` is displayed, it means the microphone is working correctly.

```commandline
python3 test_sound_device.py
```

The following problems may occur in some environments:

- Recording device was not found. As a workaround, you need to mount `/dev/snd` in a docker container, for example.
- Errors around the sound driver. Please take appropriate action according to the error you encountered.
  [example](https://stackoverflow.com/questions/49333582/portaudio-library-not-found-by-sounddevice)
- The recording itself can be done, but the acquired data is All 0.
  On Mac, this phenomenon may occur when the terminal does not have permission to use the microphone.

### Install additional functions/models (optional)

Please install additional features depending on the functionality you use.

#### Pyannote/embedding

If you want to use Pyannote to compute feature vectors for speaker identification, 
please download the model manually.

Models can be downloaded from HuggingFace.
It is a so-called gating model, 
and you need to fill out a form to register your e-mail address and other information in order to download it.
After downloaded that, place model file as  `resources/pynannote_embedding_pytorch_model.bin`.<br/>
(This article explains about a different model, but
[this](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_model.ipynb)
describes the details of procedure for downloading)

#### Blackhole (for Mac only)

[Blackhole](https://existential.audio/blackhole/) adds a virtual audio device to your Mac.
The virtual audio device has the ability to loopback the sound output as a microphone input,
so this allows the audio output of Teams, for example, to be passed directly to this application.

If you want to output to speakers and loopback simultaneously,
set up multiple output devices in the Audio MIDI settings on Mac.

### Launch the application

Run `app.py` from the terminal.

```commandline
python3 app.py
```

The first run will take some time as the model download will be performed.
`Running on local URL:  http://0.0.0.0:7860`  will be displayed when the preparation is completed.
Then, please open your Web browser and access to `http://127.0.0.1:7860/`.

### Initial setup

The application will work with the default settings, 
but we recommend changing the following settings from the "Settings" tab.

1. Specify input devices.<br/>
   Multiple inputs can be used simultaneously.
   For example, by specifying a microphone and loopback (such as Blackhole mentioned above),
   you can transcribe both the other party's voice and your own voice during a remote conference simultaneously. 
2. Settings for the device to run the recognition process;
   specifying "gpu" will cause it to use a high-precision model.
3. Setting of whether the audio data should be saved. 
   If enabled, the application will save the audio data and keep it for the specified number of days.
   This is useful to check for transcription errors later.
4. Set the algorithm for calculating the speaker feature vector. `pyannote` seems to be more accurate.
   Note that the above installation procedure must be done before launching.
5. Set the OpenAI API Key.

After you have finished all the settings, press "Apply Settings" to write the settings.


Your setup is now complete! Enjoy the app!

## Usage

Please access from your Web browser.
The address is usually `http://127.0.0.1:7860/`, 
but may be a different address or port if the application is running in a virtual machine.

You can also open the UI from another machine. 
Note that, however, the audio recording (and playback) is only performed on the machine that invoked `app.py`.

### Current tab

The current transcription is displayed in real time.
You can also check the history for the day, 
but it will automatically scroll to the bottom when an update is received.
If you want to check the details of contents, it is better to refer to the Archive tab.

### Archive tab

Transcriptions are stored by day. Select the day you want to see the history from the drop-down list.

You can also refer to the today's transcription from here,
but it will not be updated in real time on this screen.

### Diarization tab

If you have enabled the algorithm for calculating the speaker feature vectors in the Settings tab,
you can check the contents of the database here.

After a certain number of voices are collected, clustering algorithm is performed to identify persons.
A temporary name is assigned to the newly found person;
You can reassign the name by selecting the person from drop-down list, enter the new name and click "rename" button.
(If the person does not appear in the list, press "Reload").
The changed name will be reflected as much as possible, including in the history,
but not in the summary generated by GPT.

Check the "Visualize clustering results" checkbox and refresh (may take several tens of seconds),
The speaker feature vectors are compressed by t-SNE and displayed graphically.

### Settings tab

Change various settings for the application.
Some option changes require a restart of the application.

## Use remote GPU machines on the local network

Remote GPU machines on the local network can be used to process heavy DNN model processing
such as transcription and speaker feature vector computation via network communication.

After building an environment described above on the GPU machine,
run `server_main.py` on the machine. For example:

```commandline
nohup python3 server_main.py &
```

The [gRPC](https://grpc.io/) is used for RPCs.
It will be listening on port 7860 except through docker containers or virtual machines.

Once the GPU machine is ready, in the application's "Settings" tab, under "Device to perform the recognition process",
enter the access point in the format of `address:port`.


Note that VAD is always executed on local machine due to the trade-off with amount of network transmission.
Therefore, even with the settings in this section, the application itself will consume some CPU time.

## TODOs

- ~~Implement automatic language detection~~ v0.2.0
- Try an LLM that can run on local machine
- Support exporting history
- Support re-analysis of history (re-running processes after speaker identification)
- Implement capacity management of the speaker identification database
- Tuning of various parameters; We appreciate your feedback.

## License

Apache License Version 2.0
