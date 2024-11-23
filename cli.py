import argparse
from track_splitter import main as splitter


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Separate an album MP3 file into individual tracks based on detected silences."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input MP3 file."
    )
    parser.add_argument(
        "-o", "--output", default="exported_tracks", help="Output directory for the exported tracks."
    )
    parser.add_argument(
        "--min_silence_len",
        type=int,
        default=3000,
        help="Minimum length of silence to be considered a track separator, in milliseconds (default: 3000 ms).",
    )
    parser.add_argument(
        "--silence_thresh",
        type=int,
        default=None,
        help="Silence threshold in dBFS relative to the average dBFS (e.g., -16). If not set, defaults to -16.",
    )
    parser.add_argument(
        "--noise_reduce",
        action="store_true",
        help="Enable noise reduction during silence detection.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display the waveform plot with detected track boundaries.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    splitter(
        input_path=args.input,
        noise_reduce=args.noise_reduce,
        silence_thresh=args.silence_thresh,
        min_silence_len=args.min_silence_len,
        output_dir=args.output,
        display_plot=args.plot
    )


if __name__ == "__main__":
    main()
