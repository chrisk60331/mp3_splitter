import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from track_splitter import merge_close_silences, consolidate_short_tracks, process_audio, main
SOURCE_PATH = 'track_splitter'


class TestMergeCloseSilences(unittest.TestCase):
    def test_empty_input(self):
        # Given
        silences = []
        max_gap = 1000

        # When
        result = merge_close_silences(silences, max_gap)

        # Then
        self.assertEqual(result, [])

    def test_no_merge_needed(self):
        # Given
        silences = [(0, 1000), (2000, 3000), (4000, 5000)]
        max_gap = 500

        # When
        result = merge_close_silences(silences, max_gap)

        # Then
        self.assertEqual(result, silences)

    def test_merge_all(self):
        # Given
        silences = [(0, 1000), (1001, 2000), (2001, 3000)]
        max_gap = 10
        expected = [(0, 3000)]

        # When
        result = merge_close_silences(silences, max_gap)

        # Then
        self.assertEqual(result, expected)

    def test_merge_some(self):
        # Given
        silences = [(0, 1000), (1500, 2000), (3000, 3500)]
        max_gap_small = 400
        max_gap_large = 600
        expected_small_gap = [(0, 1000), (1500, 2000), (3000, 3500)]
        expected_large_gap = [(0, 2000), (3000, 3500)]

        # When & Then for small gap
        result_small_gap = merge_close_silences(silences, max_gap_small)
        self.assertEqual(result_small_gap, expected_small_gap)

        # When & Then for large gap
        result_large_gap = merge_close_silences(silences, max_gap_large)
        self.assertEqual(result_large_gap, expected_large_gap)


class TestConsolidateShortTracks(unittest.TestCase):
    def test_no_short_tracks(self):
        # Given
        tracks = [(0, 120), (120, 240), (240, 360)]
        min_duration = 60

        # When
        result = consolidate_short_tracks(tracks, min_duration)

        # Then
        self.assertEqual(result, tracks)

    def test_all_short_tracks(self):
        # Given
        tracks = [(0, 30), (30, 50), (50, 55)]
        min_duration = 60
        expected = [(0, 55)]

        # When
        result = consolidate_short_tracks(tracks, min_duration)

        # Then
        self.assertEqual(result, expected)

    def test_merge_with_next(self):
        # Given
        tracks = [(0, 50), (50, 70), (70, 130)]
        min_duration = 60
        expected = [(0, 70), (70, 130)]

        # When
        result = consolidate_short_tracks(tracks, min_duration)

        # Then
        self.assertEqual(result, expected)

    def test_merge_with_previous(self):
        # Given
        tracks = [(0, 100), (100, 150), (150, 200)]
        min_duration = 60
        expected = [(0, 100), (100, 200)]

        # When
        result = consolidate_short_tracks(tracks, min_duration)

        # Then
        self.assertEqual(result, expected)

    def test_complex_case(self):
        # Given
        tracks = [(0, 30), (30, 50), (50, 90), (90, 120), (120, 130)]
        min_duration = 60
        expected = [(0, 130)]

        # When
        result = consolidate_short_tracks(tracks, min_duration)

        # Then
        self.assertEqual(result, expected)


class TestProcessAudio(unittest.TestCase):
    @patch(f'{SOURCE_PATH}.nr.reduce_noise')
    def test_process_audio_without_noise_reduction(self, mock_reduce_noise):
        # Given
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050
        noise_reduce = False
        silence_thresh_value = -16
        min_silence_len = 3000
        min_track_duration = 60

        # When
        y_silence_detection, silence_thresh, min_silence_len_result, min_track_duration_result = process_audio(
            y, sr, noise_reduce=noise_reduce, silence_thresh_value=silence_thresh_value
        )

        # Then
        mock_reduce_noise.assert_not_called()
        self.assertTrue(np.array_equal(y_silence_detection, y))
        self.assertEqual(silence_thresh, silence_thresh_value)
        self.assertEqual(min_silence_len_result, min_silence_len)
        self.assertEqual(min_track_duration_result, min_track_duration)

    @patch(f'{SOURCE_PATH}.nr.reduce_noise')
    def test_process_audio_with_noise_reduction(self, mock_reduce_noise):
        # Given
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050
        noise_reduce = True
        silence_thresh_value = -16
        min_silence_len = 3000
        min_track_duration = 60
        mock_reduce_noise.return_value = np.array([0.05, 0.1, 0.15])

        # When
        y_silence_detection, silence_thresh, min_silence_len_result, min_track_duration_result = process_audio(
            y, sr, noise_reduce=noise_reduce, silence_thresh_value=silence_thresh_value
        )

        # Then
        mock_reduce_noise.assert_called_once()
        self.assertTrue(np.array_equal(y_silence_detection, np.array([0.05, 0.1, 0.15])))
        self.assertEqual(silence_thresh, silence_thresh_value)
        self.assertEqual(min_silence_len_result, min_silence_len)
        self.assertEqual(min_track_duration_result, min_track_duration)

    @patch(f'{SOURCE_PATH}.nr.reduce_noise')
    def test_process_audio_with_no_silence_thresh_value(self, mock_reduce_noise):
        # Given
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050
        noise_reduce = True
        silence_thresh_value = None
        min_silence_len = 3000
        min_track_duration = 60
        mock_reduce_noise.return_value = np.array([0.05, 0.1, 0.15])

        # When
        y_silence_detection, silence_thresh, min_silence_len_result, min_track_duration_result = process_audio(
            y, sr, noise_reduce=noise_reduce, silence_thresh_value=silence_thresh_value
        )

        # Then
        mock_reduce_noise.assert_called_once()
        self.assertTrue(np.array_equal(y_silence_detection, np.array([0.05, 0.1, 0.15])))
        self.assertEqual(silence_thresh, silence_thresh_value)
        self.assertEqual(min_silence_len_result, min_silence_len)
        self.assertEqual(min_track_duration_result, min_track_duration)


class TestMainFunction(unittest.TestCase):
    @patch(f'{SOURCE_PATH}.os.path.isfile')
    @patch(f'{SOURCE_PATH}.os.path.expanduser')
    @patch(f'{SOURCE_PATH}.librosa.load')
    @patch(f'{SOURCE_PATH}.librosa.get_duration')
    @patch(f'{SOURCE_PATH}.process_audio')
    @patch(f'{SOURCE_PATH}.AudioSegment.from_wav')
    @patch(f'{SOURCE_PATH}.AudioSegment.from_file')
    @patch(f'{SOURCE_PATH}.silence.detect_silence')
    @patch(f'{SOURCE_PATH}.merge_close_silences')
    @patch(f'{SOURCE_PATH}.consolidate_short_tracks')
    @patch(f'{SOURCE_PATH}.os.makedirs')
    def test_main(
        self,
        mock_makedirs,
        mock_consolidate_short_tracks,
        mock_merge_close_silences,
        mock_detect_silence,
        mock_from_file,
        mock_from_wav,
        mock_process_audio,
        mock_get_duration,
        mock_load,
        mock_expanduser,
        mock_isfile
    ):
        # Given
        input_path = 'test.mp3'
        output_dir = 'output'
        noise_reduce = False
        silence_thresh = -16
        min_silence_len = 3000
        display_plot = False

        # Mock configurations
        mock_isfile.return_value = True
        mock_expanduser.side_effect = lambda x: x
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050
        duration = 180.0
        y_silence_detection = y
        min_track_duration = 60
        silences_detected = [(0, 1000), (2000, 3000)]
        merged_silences = [(0, 1000), (2000, 3000)]
        consolidated_tracks = [(0, 180000)]
        audio_segment_mock = MagicMock()

        mock_load.return_value = (y, sr)
        mock_get_duration.return_value = duration
        mock_process_audio.return_value = (y_silence_detection, silence_thresh, min_silence_len, min_track_duration)
        mock_detect_silence.return_value = silences_detected
        mock_merge_close_silences.return_value = merged_silences
        mock_consolidate_short_tracks.return_value = consolidated_tracks
        mock_from_file.return_value = audio_segment_mock
        mock_from_wav.return_value = audio_segment_mock

        # When
        main(
            input_path=input_path,
            noise_reduce=noise_reduce,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            output_dir=output_dir,
            display_plot=display_plot
        )

        # Then
        mock_isfile.assert_called_once_with(input_path)
        mock_load.assert_called_once_with(input_path, sr=None)
        mock_get_duration.assert_called_once_with(y=y, sr=sr)
        mock_process_audio.assert_called_once_with(
            y, sr, noise_reduce=noise_reduce, silence_thresh_value=silence_thresh,
            min_silence_len=min_silence_len, min_track_duration=60
        )
        mock_detect_silence.assert_called_once()
        mock_merge_close_silences.assert_called_once_with(silences_detected, max_gap=1000)
        mock_makedirs.assert_called_with(output_dir, exist_ok=True)
        self.assertTrue(audio_segment_mock.__getitem__.called)
        self.assertTrue(audio_segment_mock.__getitem__().export.called)

    def test_main_no_file(
        self,
    ):
        # Given
        input_path = 'doesnt_exist.mp3'
        output_dir = 'output'
        noise_reduce = False
        silence_thresh = -16
        min_silence_len = 3000
        display_plot = False

        # When
        main(
            input_path=input_path,
            noise_reduce=noise_reduce,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            output_dir=output_dir,
            display_plot=display_plot
        )

    @patch(f'{SOURCE_PATH}.AudioSegment.from_file')
    @patch(f'{SOURCE_PATH}.librosa.load')
    @patch(f'{SOURCE_PATH}.os.path.isfile')
    @patch(f'{SOURCE_PATH}.os.path.expanduser')
    def test_main_no_silence_thresh(
        self,
        mock_expanduser,
        mock_isfile,
        mock_load,
        mock_from_file,
    ):
        # Given
        input_path = 'test.mp3'
        output_dir = 'output'
        noise_reduce = False
        silence_thresh = None
        min_silence_len = 3000
        display_plot = False
        mock_isfile.return_value = True
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050

        mock_load.return_value = (y, sr)
        # When
        main(
            input_path=input_path,
            noise_reduce=noise_reduce,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            output_dir=output_dir,
            display_plot=display_plot
        )

    @patch(f'{SOURCE_PATH}.os.path.isfile')
    @patch(f'{SOURCE_PATH}.os.path.expanduser')
    @patch(f'{SOURCE_PATH}.librosa.load')
    @patch(f'{SOURCE_PATH}.librosa.get_duration')
    @patch(f'{SOURCE_PATH}.process_audio')
    @patch(f'{SOURCE_PATH}.AudioSegment.from_wav')
    @patch(f'{SOURCE_PATH}.AudioSegment.from_file')
    @patch(f'{SOURCE_PATH}.silence.detect_silence')
    @patch(f'{SOURCE_PATH}.merge_close_silences')
    @patch(f'{SOURCE_PATH}.consolidate_short_tracks')
    @patch(f'{SOURCE_PATH}.os.makedirs')
    @patch(f'{SOURCE_PATH}.plt')
    def test_main_plot(
            self,
            mock_plt,
            mock_makedirs,
            mock_consolidate_short_tracks,
            mock_merge_close_silences,
            mock_detect_silence,
            mock_from_file,
            mock_from_wav,
            mock_process_audio,
            mock_get_duration,
            mock_load,
            mock_expanduser,
            mock_isfile
    ):
        # Given
        input_path = 'test.mp3'
        output_dir = 'output'
        noise_reduce = False
        silence_thresh = -16
        min_silence_len = 3000
        display_plot = True

        # Mock configurations
        mock_isfile.return_value = True
        mock_expanduser.side_effect = lambda x: x
        y = np.array([0.1, 0.2, 0.3])
        sr = 22050
        duration = 180.0
        y_silence_detection = y
        min_track_duration = 60
        silences_detected = [(0, 1000), (2000, 3000)]
        merged_silences = [(0, 1000), (2000, 3000)]
        consolidated_tracks = [(0, 180000)]
        audio_segment_mock = MagicMock()

        mock_load.return_value = (y, sr)
        mock_get_duration.return_value = duration
        mock_process_audio.return_value = (y_silence_detection, silence_thresh, min_silence_len, min_track_duration)
        mock_detect_silence.return_value = silences_detected
        mock_merge_close_silences.return_value = merged_silences
        mock_consolidate_short_tracks.return_value = consolidated_tracks
        mock_from_file.return_value = audio_segment_mock
        mock_from_wav.return_value = audio_segment_mock

        # When
        main(
            input_path=input_path,
            noise_reduce=noise_reduce,
            silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            output_dir=output_dir,
            display_plot=display_plot
        )
