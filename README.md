# Python mp3 Track Splitter


## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python cli.py -i "~/Music/Music/Media.localized/Music/Teddy.Swims/album.mp3" --min_silence_len 4000 --silence_thresh -18
```

## Tests
```bash
pytest -cov tests
```