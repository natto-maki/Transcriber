"""
Check the operation of the sound device

When running PyCharm from JetBrains ToolBox on Mac,
it may not be able to record sound because permissions are not granted.
To solve this situation, for example, you can grant permissions to the terminal and start PyCharm from the terminal.

  cd /Users/<user>/Applications/PyCharm\ Community\ Edition.app/Contents/MacOS
  ./pycharm
"""
import sounddevice as sd
import numpy as np
import sys

fs = 16000
duration = 5
rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
print("pass" if np.max(rec) > sys.float_info.epsilon else "failed")
