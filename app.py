import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List
import io
import soundfile as sf
import timeit

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client

from streamlit_webrtc import WebRtcMode, webrtc_streamer

import whisper

import speech_recognition as sr


HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too, 
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """
    # testing123
    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers


def main():

    st.header("Real Time Speech-to-Text")

    if True:
        app_sst()


def app_sst():
    whisper_model = whisper.load_model('base.en')
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024*10,  # 1024
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    whisper_output_container = st.empty()
    whisper_output = []
    adjusted_for_ambient_noise = True
    first_30_seconds = True
    first_run = True
    r = sr.Recognizer()
    debugger = st.empty()
    whisper_transcribe_frequency = 3
    buffer_whisper_output = ""
    final_whisper_output = ["History of present illness:"]

    start_time = time.time()

    whisper_initial_prompt = """
    The following is a dictation by an Emergency Physician who is describing the past medical history, history of present illness, symptoms, physical exam, impression and plan as it pertains to a given patient.
    """

    while True:
        if webrtc_ctx.audio_receiver:

            if not adjusted_for_ambient_noise:
                status_indicator.write('Calibrating for ambient noise')
                time.sleep(3)
            else:
                status_indicator.write("Listening")

            if first_run:
                recording = pydub.AudioSegment.empty()
                first_run = False
                time.sleep(3)

            # sound_chunk = pydub.AudioSegment.empty()

            time.sleep(1)

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                recording += sound

            debugger.write(len(recording))

            if len(recording) > 0 and len(recording) < 30000:
                last_cutoff = len(recording)
                recording = recording.set_channels(1).set_frame_rate(
                    16000
                )
                sample_width = audio_frames[0].format.bytes
                sample_rate = audio_frames[0].sample_rate
                recording = recording.set_channels(
                    1).set_frame_rate(sample_rate)

                buffer = np.array((recording).get_array_of_samples())
                # buffer = np.array(
                #         sound_chunk[-30000:].get_array_of_samples())

                logger.info(
                    f"trying to recognize {len(recording)} chunks with rate:{sample_width} width:{sample_rate}"
                )

                whisper_audio = sr.AudioData(buffer, sample_rate, sample_width)

                wav_bytes = whisper_audio.get_wav_data(convert_rate=16000)
                wav_stream = io.BytesIO(wav_bytes)
                audio_array, sampling_rate = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)

                # start = timeit.default_timer()
                result = whisper_model.transcribe(
                    audio_array, no_speech_threshold=0.5, initial_prompt="\n".join(final_whisper_output))
                # stop = timeit.default_timer()
                # execution_time = stop - start

                if result:
                    buffer_whisper_output = result['text']

                    # whisper_output.append(result['text'])
                whisper_output_container.write(
                    final_whisper_output + [buffer_whisper_output])

            if len(recording) > 30000:
                final_whisper_output.append(buffer_whisper_output)
                recording = recording[last_cutoff:]

        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in [
        "false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
