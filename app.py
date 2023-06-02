import gradio as gr
import subprocess
import os
import webvtt
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()


def download_audio(url, output_dir):
    # Use subprocess to call yt-dlp to download the video and convert it to an MP3 file
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format",
            "mp3",
            "-o",
            f"{output_dir}/audio.%(ext)s",
            url,
        ]
    )


def transcribe_audio(output_dir, min_speakers=1, max_speakers=2):
    # Use subprocess to call whisperX model to transcribe the audio
    subprocess.run(
        [
            "whisperx",
            f"{output_dir}/audio.mp3",
            "--model",
            "large-v2",
            "--diarize",
            "--min_speakers",
            f"{int(min_speakers)}",
            "--max_speakers",
            f"{int(max_speakers)}",
            "--output_format",
            "vtt",
            "--output_dir",
            f"{output_dir}",
            "--hf_token",
            f"{os.environ.get('HF_TOKEN')}",
        ]
    )

def import_vtt_file(vtt_file):
    # Use the webvtt library to read the .vtt file
    captions = webvtt.read(vtt_file)

    # Print out the text of each caption
    return "\n".join([caption.text for caption in captions])

def download_transcribe_audio_gradio(url,*args,  output_dir="output"):
    # Check if the output directory exists
    if not os.path.exists(output_dir):
        # If it doesn't exist, create it
        os.makedirs(output_dir)

    # Call the download_audio function
    download_audio(url, output_dir)

    # Call the transcribe_audio function
    transcribe_audio(output_dir, *args)

    # Remove the audio file
    os.remove(f"{output_dir}/audio.mp3")

    # Return the status
    return import_vtt_file(f"{output_dir}/audio.vtt")


# Create the input and output interfaces
inputs = [
    gr.inputs.Textbox(default="https://www.youtube.com/watch?v=FKxa-WQd3Z8&t", label="Enter the URL of the YouTube audio you want to download:"),
    gr.inputs.Number(default=1, label="Enter the minimum number of speakers:"),
    gr.inputs.Number(default=2, label="Enter the maximum number of speakers:"),
]

output = gr.outputs.Textbox(label='Transcription:')

# Create the Gradio app
app = gr.Interface(
    download_transcribe_audio_gradio,
    inputs,
    output,
    title="YouTube Audio Transcriber with Diarization",
    description="Transcribe YouTube audio with the whisperX model. Provide the URL of the YouTube audio, and the minimum and maximum number of speakers you want to annotate.",
)

# Run the app
app.launch(debug=True)
