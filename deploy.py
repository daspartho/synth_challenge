from cerebrium import Conduit, model_type, hardware
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

import base64
# write a function to convert base64 to audio file
def pre_process(data):
    with open("audio.flac", "wb") as f:
        f.write(base64.b64decode(data))
    return 'audio.flac'

# Create a conduit
c = Conduit(
    'test',
    API_KEY,
    [
        (model_type.HUGGINGFACE_PIPELINE, {"task": "automatic-speech-recognition", "model": "nikhilbh/whisper-medium-custom-hi"}, {"pre": pre_process}),
    ],
)

c.deploy()