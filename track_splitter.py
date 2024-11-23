import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, silence
from scipy.io.wavfile import write


def merge_close_silences(silences, max_gap=1000):
    if not silences:
        return []
    merged_silences = [silences[0]]
    for current in silences[1:]:
        previous = merged_silences[-1]
        if current[0] - previous[1] <= max_gap:
            # Merge the two silences
            merged_silences[-1] = (previous[0], current[1])
        else:
            merged_silences.append(current)
    return merged_silences


def process_audio(
    y,
    sr,
    noise_reduce=False,
    silence_thresh_value=None,
    min_silence_len=3000,
    min_track_duration=60
):
    # Noise reduction (for silence detection only)
    if noise_reduce:
        # Estimate noise sample (adjust threshold as needed)
        noise_sample = y[np.abs(y) < 0.02]
        # Reduce noise
        y_reduced = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=1.0)
        y_silence_detection = y_reduced
    else:
        y_silence_detection = y

    # Prepare audio for silence detection (simulate pydub.AudioSegment behavior)
    # Since pydub works with AudioSegment, we'll need to mock this in tests

    # Adjusted silence detection parameters
    if silence_thresh_value is None:
        # Default value
        silence_thresh = None  # To be set after loading into AudioSegment
    else:
        silence_thresh = silence_thresh_value

    # Return necessary data for further processing
    return y_silence_detection, silence_thresh, min_silence_len, min_track_duration


def consolidate_short_tracks(tracks, min_duration):
    consolidated_tracks = []
    i = 0
    while i < len(tracks):
        start, end = tracks[i]
        duration = end - start

        if duration >= min_duration:
            consolidated_tracks.append((start, end))
            i += 1
        else:
            # Merge with the next track(s) until duration is sufficient
            merge_end = end
            while duration < min_duration and i + 1 < len(tracks):
                i += 1
                merge_end = tracks[i][1]
                duration = merge_end - start
            # If still too short and there is a previous track, merge with it
            if duration < min_duration and consolidated_tracks:
                prev_start, _ = consolidated_tracks[-1]
                consolidated_tracks[-1] = (prev_start, merge_end)
            else:
                consolidated_tracks.append((start, merge_end))
            i += 1
    return consolidated_tracks


def main(
    input_path=None,
    noise_reduce=None,
    silence_thresh=None,
    min_silence_len=None,
    output_dir=None,
    display_plot=None
):
    # Check if input file exists
    input_path = os.path.expanduser(input_path)

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    # Load the original audio file with librosa
    y, sr = librosa.load(input_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds")

    y_silence_detection, silence_thresh_value, min_silence_len, min_track_duration = process_audio(
        y,
        sr,
        noise_reduce=noise_reduce,
        silence_thresh_value=silence_thresh,
        min_silence_len=min_silence_len or 3000,
        min_track_duration=60  # Or make this an argument
    )

    # Save the audio used for silence detection to a temporary WAV file (for pydub compatibility)
    temp_wav_path = "temp_silence_detection.wav"
    write(temp_wav_path, sr, (y_silence_detection * 32767).astype(np.int16))  # Convert to 16-bit PCM

    # Load the audio used for silence detection with pydub
    audio_for_detection = AudioSegment.from_wav(temp_wav_path)

    # Load the original audio with pydub (for exporting tracks)
    audio_original = AudioSegment.from_file(input_path)

    # Adjusted silence detection parameters
    if silence_thresh is None:
        silence_thresh = audio_for_detection.dBFS - 16  # Default value
    else:
        silence_thresh = audio_for_detection.dBFS + silence_thresh  # User-defined relative value

    # Detect silent intervals using the audio prepared for silence detection
    print("Detecting silences...")
    silent_ranges = silence.detect_silence(
        audio_for_detection,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=100
    )

    # Remove the temporary WAV file
    os.remove(temp_wav_path)

    # Merge close silences
    silent_ranges = merge_close_silences(silent_ranges, max_gap=1000)

    # Convert silent_ranges from ms to seconds
    silent_ranges_sec = [(start / 1000.0, stop / 1000.0) for start, stop in silent_ranges]
    print("\nDetected silent intervals (in seconds):")
    for start, stop in silent_ranges_sec:
        print(f"Silence from {start:.2f}s to {stop:.2f}s")

    # Build the initial list of tracks
    raw_tracks = []
    for i in range(len(silent_ranges_sec) + 1):
        if i == 0:
            start = 0.0
            end = silent_ranges_sec[0][0] if silent_ranges_sec else duration
        elif i == len(silent_ranges_sec):
            start = silent_ranges_sec[-1][1]
            end = duration
        else:
            start = silent_ranges_sec[i - 1][1]
            end = silent_ranges_sec[i][0]
        raw_tracks.append((start, end))

    # Consolidate short tracks
    MIN_TRACK_DURATION = 60  # Minimum track duration in seconds
    tracks = consolidate_short_tracks(raw_tracks, MIN_TRACK_DURATION)

    # Print track boundaries after consolidation
    print("\nDetected track boundaries (after consolidation):")
    for idx, (start, end) in enumerate(tracks, 1):
        duration = end - start
        print(f"Track {idx}: {start:.2f}s to {end:.2f}s (Duration: {duration:.2f}s)")

    # Create an output directory for the exported tracks
    os.makedirs(output_dir, exist_ok=True)

    # Export each track as a separate MP3 file (using original audio)
    print("\nExporting tracks...")
    for idx, (start, end) in enumerate(tracks, 1):
        start_ms = start * 1000
        end_ms = end * 1000
        track_audio = audio_original[start_ms:end_ms]
        output_filename = f"Track_{idx:02d}.mp3"
        output_path = os.path.join(output_dir, output_filename)
        track_audio.export(output_path, format="mp3")
        print(f"Exported Track {idx}: {output_filename} ({start:.2f}s to {end:.2f}s)")

    # Update plotting code to display consolidated tracks
    if display_plot:
        print("\nDisplaying waveform plot...")
        plt.figure(figsize=(20, 6))
        librosa.display.waveshow(y_silence_detection, sr=sr, alpha=0.6)
        plt.title("Album Waveform with Consolidated Track Boundaries")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Add vertical lines for consolidated track boundaries
        for start, end in tracks:
            plt.axvline(x=start, color='green', linestyle='--', alpha=0.8)
            plt.axvline(x=end, color='green', linestyle='--', alpha=0.8)

        # Label tracks
        for idx, (start, end) in enumerate(tracks, 1):
            plt.text((start + end) / 2, np.max(y_silence_detection), f'Track {idx}',
                     horizontalalignment='center', color='blue', fontsize=9, rotation=90)

        plt.tight_layout()
        plt.show()
