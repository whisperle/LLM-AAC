# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import time
import io
import h5py
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from fairseq.data import FairseqDataset
from ..utils.data_utils import compute_block_mask_1d, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)



class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask=False,
        feature_encoder_spec: str = "None",
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        corpus_key=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

        self.is_compute_mask = compute_mask
        self.feature_encoder_spec = eval(feature_encoder_spec)
        self._features_size_map = {}
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.corpus_key = corpus_key

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)


    def postprocess(self, feats, curr_sample_rate, roll_aug = False):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()
        # if self.normalize:
        #     with torch.no_grad():
        #         feats = F.layer_norm(feats, feats.shape)
        feats = feats - feats.mean()
        
        if roll_aug:
            feats = self._roll_mag_aug(feats)
        
        return feats

    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.corpus_key is not None:
            input["corpus_key"] = [self.corpus_key] * len(sources)
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self.crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self.feature_encoder_spec:
            L_in = size
            for (_, kernel_size, stride) in self.feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        h5_format=False,
        downsr_16hz=False,
        wav2fbank=False,
        target_length=1024,
        esc50_eval=False,
        spcv2_eval=False,
        roll_mag_aug=False,
        noise=False,
        train_mode='train',
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)
        self.h5_format = h5_format
        self.downsr_16hz = downsr_16hz
        self.wav2fbank = wav2fbank
        self.target_length = target_length
        self.esc50_eval = esc50_eval
        self.spcv2_eval = spcv2_eval
        self.roll_mag_aug = roll_mag_aug
        self.noise = noise
        self.train_mode = train_mode

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        # exclude data not in sample rate range     10.h5/****.wav  320000 
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split()
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        if self.esc50_eval:
            task_dataset = "ESC-50"
        elif self.spcv2_eval:
            task_dataset = "SPC-2"
        else:
            task_dataset = "AS"
        
        logger.info(
            f"sample rate: 16000\t"
            f"target length: {self.target_length}\t"
            f"current task: {task_dataset}\t"
        )

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)
        # print("skipped_index: {}".format(self.skipped_indices))
        # print(len(self.skipped_indices))

    # two file format. h5_format = true -> .h5(.hdf5) ; h5_format = false -> .wav
    def __getitem__(self, index):
        import soundfile as sf

        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])   # root/10.h5/***.wav
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        retry = 3
        wav = None
        for i in range(retry):
            try:
                if self.h5_format and self.train_mode == 'train':
                    parts = path_or_fp.split("/")
                    path_or_fp = "/".join(parts[:-1])
                    path_or_fp = h5py.File(path_or_fp,'r')
                    wav = path_or_fp[parts[-1]][:]
                    curr_sample_rate = 32000
                    break                    
                else:
                    wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
                    break
            except Exception as e:
                logger.warning(
                    f"Failed to read {path_or_fp}: {e}. Sleeping for {1 * i}"
                )
                time.sleep(1 * i)

        if wav is None:
            raise Exception(f"Failed to load {path_or_fp}")

        if self.h5_format:
            feats = torch.tensor(wav).float()
        else:
            feats = torch.from_numpy(wav).float()
            
        if self.downsr_16hz:
            feats = torchaudio.functional.resample(feats, orig_freq=curr_sample_rate, new_freq=16000)
            curr_sample_rate = 16000
            self.sample_rate = curr_sample_rate
            
        # whether to use roll augmentation on waveform
        use_roll = self.roll_mag_aug and self.train_mode == 'train'
        
        feats = self.postprocess(feats, curr_sample_rate, use_roll)

        # convert waveform to spectrogram
        if self.wav2fbank:
            feats = feats.unsqueeze(dim=0)
            feats = torchaudio.compliance.kaldi.fbank(feats, htk_compat=True, sample_frequency=curr_sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
            
            # padding 
            n_frames = feats.shape[1]
            diff = self.target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                feats = m(feats)
                
            elif diff < 0:
                feats = feats[:,0:self.target_length,:]     
                
            # global normalization for AS
            self.norm_mean = -4.268 
            self.norm_std = 4.569
            
            # global normalization for ESC-50
            if self.esc50_eval:
                self.norm_mean = -6.627
                self.norm_std = 5.359
                
            # global normalization for spcv2
            if self.spcv2_eval:
                self.norm_mean = -6.846
                self.norm_std = 5.565
                
            feats = (feats - self.norm_mean) / (self.norm_std * 2) 
            
            if self.noise and self.train_mode == 'train': 
                feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
                feats = torch.roll(feats, np.random.randint(-10, 10), 1)

        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v

class HuggingFaceAudioDataset(RawAudioDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        wav2fbank=False,
        target_length=1024,
        esc50_eval=False,
        spcv2_eval=False,
        roll_mag_aug=False,
        noise=False,
        train_mode='train',
        dataset_config="balanced",  # "balanced" or "unbalanced"
        cache_dir=None,
        feature_type="melspec",  # "melspec" or "cwt"
        cwt_scales=64,  # Number of scales for CWT
        cwt_width=8.0,  # Width parameter for Morlet wavelet
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )
        
        self.wav2fbank = wav2fbank
        self.target_length = target_length
        self.esc50_eval = esc50_eval
        self.spcv2_eval = spcv2_eval
        self.roll_mag_aug = roll_mag_aug
        self.noise = noise
        self.train_mode = train_mode
        self.feature_type = feature_type
        self.cwt_scales = cwt_scales
        self.cwt_width = cwt_width
        
        # Use default cache directory if none provided
        if cache_dir is None:
            cache_dir = "/scratch/cl6707/Shared_Datasets/AudioSet"
            # Create the directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
        # Import here to avoid adding an unnecessary dependency if not using this class
        from datasets import load_dataset
        
        # Check if PyWavelets is installed if using CWT
        if self.feature_type == "cwt":
            try:
                import pywt
            except ImportError:
                logger.warning("PyWavelets not found. Installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "PyWavelets"])
        
        # Load HuggingFace dataset
        self.hf_dataset = load_dataset("agkphysics/AudioSet", dataset_config, cache_dir=cache_dir)
        
        # Map split name from EAT convention to HuggingFace convention
        hf_split = "train" if train_mode == "train" else "test"
        self.dataset = self.hf_dataset[hf_split]
        
        # Calculate sizes similar to FileAudioDataset
        skipped = 0
        sizes = []
        self.skipped_indices = set()
        
        for i in range(len(self.dataset)):
            sz = len(self.dataset[i]["audio"]["array"])
            if min_sample_size is not None and sz < min_sample_size:
                skipped += 1
                self.skipped_indices.add(i)
                continue
            sizes.append(sz)
        
        logger.info(f"loaded {len(sizes)}, skipped {skipped} samples")
        
        if self.esc50_eval:
            task_dataset = "ESC-50"
        elif self.spcv2_eval:
            task_dataset = "SPC-2"
        else:
            task_dataset = "AS"
        
        logger.info(
            f"sample rate: {sample_rate}\t"
            f"target length: {self.target_length}\t"
            f"current task: {task_dataset}\t"
            f"feature type: {self.feature_type}\t"
        )
        
        self.sizes = np.array(sizes, dtype=np.int64)
        self.set_bucket_info(num_buckets)
    
    def compute_cwt(self, waveform, scales=64, width=8.0, max_len=1024):
        """
        Compute Continuous Wavelet Transform using Morlet wavelet
        
        Args:
            waveform: Audio waveform (1D tensor)
            scales: Number of scales for the CWT
            width: Width parameter for Morlet wavelet
            max_len: Maximum length of the output (time dimension)
            
        Returns:
            CWT coefficient matrix as torch tensor
        """
        import pywt
        import numpy as np
        
        # Ensure reasonable data length to prevent memory issues
        data_len = len(waveform)
        if data_len > 160000:  # Cap at ~10 seconds for 16kHz audio
            waveform = waveform[:160000]
            data_len = 160000
        
        # Convert to numpy
        signal = waveform.numpy()
        
        # Create scales (frequency bands) - using a logarithmic scale similar to mel-spectrogram
        scales = np.logspace(np.log10(4), np.log10(data_len // 2), num=scales)
        
        # Compute CWT with Morlet wavelet
        coef, _ = pywt.cwt(signal, scales, 'morl', 1.0 / self.sample_rate)
        
        # Use absolute value (magnitude) of the complex CWT
        coef = np.abs(coef)
        
        # Reshape to target length through interpolation
        if max_len is not None:
            from scipy import signal as sps
            # Resample the time dimension to target_length
            if coef.shape[1] != max_len:
                coef = sps.resample(coef, max_len, axis=1)
        
        # Convert back to tensor and add batch dimension
        coef_tensor = torch.from_numpy(coef).float().unsqueeze(0)
        
        return coef_tensor
    
    def __getitem__(self, index):
        # Get audio from HuggingFace dataset
        sample = self.dataset[index]
        audio_data = sample["audio"]["array"]
        curr_sample_rate = sample["audio"]["sampling_rate"]
        
        # Convert to torch tensor
        feats = torch.tensor(audio_data).float()
        
        # Resample to target sample rate if needed
        if curr_sample_rate != self.sample_rate:
            feats = torchaudio.functional.resample(feats, orig_freq=curr_sample_rate, new_freq=self.sample_rate)
            curr_sample_rate = self.sample_rate
        
        # whether to use roll augmentation on waveform
        use_roll = self.roll_mag_aug and self.train_mode == 'train'
        
        feats = self.postprocess(feats, curr_sample_rate, use_roll)
        
        # If using CWT
        if self.feature_type == "cwt" and self.wav2fbank:
            feats = self.compute_cwt(
                feats, 
                scales=self.cwt_scales, 
                width=self.cwt_width, 
                max_len=self.target_length
            )
            
            # Apply normalization based on dataset
            if self.esc50_eval:
                self.norm_mean = -0.5  # These are placeholder values - need to be computed for CWT
                self.norm_std = 1.0
            elif self.spcv2_eval:
                self.norm_mean = -0.5  # These are placeholder values - need to be computed for CWT
                self.norm_std = 1.0
            else:  # AudioSet
                self.norm_mean = -0.5  # These are placeholder values - need to be computed for CWT
                self.norm_std = 1.0
            
            # Normalize
            feats = (feats - self.norm_mean) / (self.norm_std * 2)
            
            # Apply noise if needed (similar to mel-spec version)
            if self.noise and self.train_mode == 'train':
                feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
                feats = torch.roll(feats, np.random.randint(-10, 10), 1)
        
        # convert waveform to spectrogram (original mel-spec implementation)
        elif self.wav2fbank:
            feats = feats.unsqueeze(dim=0)
            feats = torchaudio.compliance.kaldi.fbank(feats, htk_compat=True, sample_frequency=curr_sample_rate, 
                                                     use_energy=False, window_type='hanning', 
                                                     num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
            
            # padding 
            n_frames = feats.shape[1]
            diff = self.target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                feats = m(feats)
            elif diff < 0:
                feats = feats[:,0:self.target_length,:]     
            
            # global normalization for AS
            self.norm_mean = -4.268 
            self.norm_std = 4.569
            
            # global normalization for ESC-50
            if self.esc50_eval:
                self.norm_mean = -6.627
                self.norm_std = 5.359
                
            # global normalization for spcv2
            if self.spcv2_eval:
                self.norm_mean = -6.846
                self.norm_std = 5.565
                
            feats = (feats - self.norm_mean) / (self.norm_std * 2) 
            
            if self.noise and self.train_mode == 'train': 
                feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
                feats = torch.roll(feats, np.random.randint(-10, 10), 1)
        
        v = {"id": index, "source": feats}
        
        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )
            
            v["precomputed_mask"] = mask
            
        return v

class PrecomputedCWTAudioDataset(RawAudioDataset):
    """
    Dataset for loading pre-computed CWT features for AudioSet.
    This eliminates the need to compute CWT on-the-fly.
    """
    
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        features_path=None,
        config="balanced",
        split="train",
        target_length=1024,
        esc50_eval=False,
        spcv2_eval=False,
        roll_mag_aug=False,
        noise=False,
        train_mode='train',
        force_compute=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )
        
        self.target_length = target_length
        self.esc50_eval = esc50_eval
        self.spcv2_eval = spcv2_eval
        self.roll_mag_aug = roll_mag_aug
        self.noise = noise
        self.train_mode = train_mode
        self.config = config
        self.split = split
        
        # Set default features path if none provided
        if features_path is None:
            features_path = f"/scratch/cl6707/Shared_Datasets/AudioSet_Wavelets/latest_{config}_{split}_cwt_features.h5"
        
        self.features_path = features_path
        
        # Check if precomputed features exist
        if not os.path.exists(features_path) or force_compute:
            logger.warning(f"Pre-computed features file not found at {features_path} or force_compute=True.")
            logger.warning("Falling back to HuggingFace dataset with on-the-fly computation.")
            
            # Use HuggingFace dataset with on-the-fly computation
            from datasets import load_dataset
            
            self.using_precomputed = False
            self.hf_dataset = load_dataset("agkphysics/AudioSet", config)
            self.dataset = self.hf_dataset[split if split != "train" else "train"]
            
            # Calculate sizes
            skipped = 0
            sizes = []
            self.skipped_indices = set()
            
            for i in range(len(self.dataset)):
                sz = len(self.dataset[i]["audio"]["array"])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                sizes.append(sz)
            
            logger.info(f"Using HuggingFace dataset: loaded {len(sizes)}, skipped {skipped} samples")
            self.video_ids = [self.dataset[i]["video_id"] for i in range(len(self.dataset)) if i not in self.skipped_indices]
            self.valid_indices = [i for i in range(len(self.dataset)) if i not in self.skipped_indices]
            
            # Default normalization values for AudioSet CWT features
            self.norm_mean = -0.5  # Placeholder
            self.norm_std = 1.0    # Placeholder
            
        else:
            # Load metadata from pre-computed features
            try:
                self.using_precomputed = True
                self.h5_file = h5py.File(features_path, 'r')
                
                # Get normalization values from file attributes or stats
                if 'global_mean' in self.h5_file.attrs and 'global_std' in self.h5_file.attrs:
                    self.norm_mean = float(self.h5_file.attrs['global_mean'])
                    self.norm_std = float(self.h5_file.attrs['global_std'])
                    logger.info(f"Using global stats from HDF5 attributes: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}")
                else:
                    # Try to load from stats JSON
                    stats_dir = os.path.join(os.path.dirname(features_path), config, "stats")
                    stats_file = os.path.join(stats_dir, f"{split}_cwt_stats.json")
                    
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                            self.norm_mean = float(stats["global_mean"])
                            self.norm_std = float(stats["global_std"])
                        logger.info(f"Using global stats from JSON: mean={self.norm_mean:.4f}, std={self.norm_std:.4f}")
                    else:
                        # Use placeholder values
                        self.norm_mean = -0.5
                        self.norm_std = 1.0
                        logger.warning("Could not find normalization stats. Using placeholder values.")
                
                # Get feature indices and metadata
                self.features_group = self.h5_file['features']
                self.metadata_group = self.h5_file['metadata']
                
                # Get valid indices (those that have features)
                self.valid_indices = sorted([int(k) for k in self.features_group.keys()])
                self.skipped_indices = set()
                
                # Load metadata for each valid index
                self.video_ids = []
                for idx in self.valid_indices:
                    metadata_bytes = self.metadata_group[str(idx)][()]
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                    self.video_ids.append(metadata["video_id"])
                
                logger.info(f"Using pre-computed features: loaded {len(self.valid_indices)} samples")
                
            except Exception as e:
                logger.error(f"Error loading pre-computed features: {str(e)}")
                logger.warning("Falling back to HuggingFace dataset with on-the-fly computation.")
                
                # Fallback to HuggingFace with on-the-fly computation
                from datasets import load_dataset
                
                self.using_precomputed = False
                self.hf_dataset = load_dataset("agkphysics/AudioSet", config)
                self.dataset = self.hf_dataset[split if split != "train" else "train"]
                
                # Calculate sizes
                skipped = 0
                sizes = []
                self.skipped_indices = set()
                
                for i in range(len(self.dataset)):
                    sz = len(self.dataset[i]["audio"]["array"])
                    if min_sample_size is not None and sz < min_sample_size:
                        skipped += 1
                        self.skipped_indices.add(i)
                        continue
                    sizes.append(sz)
                
                logger.info(f"Using HuggingFace dataset: loaded {len(sizes)}, skipped {skipped} samples")
                self.video_ids = [self.dataset[i]["video_id"] for i in range(len(self.dataset)) if i not in self.skipped_indices]
                self.valid_indices = [i for i in range(len(self.dataset)) if i not in self.skipped_indices]
                
                # Default normalization values
                self.norm_mean = -0.5
                self.norm_std = 1.0
        
        # Calculate sizes based on valid indices
        if self.using_precomputed:
            # For pre-computed features, size is fixed
            self.sizes = np.full(len(self.valid_indices), self.target_length, dtype=np.int64)
        else:
            # For on-the-fly computation, use audio sizes
            sizes = []
            for idx in self.valid_indices:
                sizes.append(len(self.dataset[idx]["audio"]["array"]))
            self.sizes = np.array(sizes, dtype=np.int64)
        
        if self.esc50_eval:
            task_dataset = "ESC-50"
        elif self.spcv2_eval:
            task_dataset = "SPC-2"
        else:
            task_dataset = "AS"
        
        logger.info(
            f"Target length: {self.target_length}\t"
            f"Current task: {task_dataset}\t"
            f"Using precomputed features: {self.using_precomputed}\t"
        )
        
        self.set_bucket_info(num_buckets)
    
    def compute_cwt(self, waveform, sample_rate=16000, scales=64, width=8.0, max_len=1024):
        """
        Compute Continuous Wavelet Transform using Morlet wavelet
        Only used as fallback if pre-computed features are not available
        """
        try:
            import pywt
        except ImportError:
            logger.warning("PyWavelets not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyWavelets"])
            import pywt
        
        try:
            from scipy import signal as sps
        except ImportError:
            logger.warning("SciPy not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
            from scipy import signal as sps
        
        # Ensure input is numpy array
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Ensure reasonable data length to prevent memory issues
        data_len = len(waveform)
        if data_len > 160000:  # Cap at ~10 seconds for 16kHz audio
            waveform = waveform[:160000]
            data_len = 160000
        
        # Create scales (frequency bands) - using a logarithmic scale
        scales = np.logspace(np.log10(4), np.log10(data_len // 2), num=scales)
        
        # Compute CWT with Morlet wavelet
        coef, _ = pywt.cwt(waveform, scales, 'morl', 1.0 / sample_rate)
        
        # Use absolute value (magnitude) of the complex CWT
        coef = np.abs(coef)
        
        # Reshape to target length through interpolation
        if max_len is not None and coef.shape[1] != max_len:
            coef = sps.resample(coef, max_len, axis=1)
        
        return coef
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map idx to actual index in the dataset
        index = self.valid_indices[idx]
        
        if self.using_precomputed:
            # Get pre-computed features
            try:
                cwt_features = self.features_group[str(index)][:]
                
                # Convert to torch tensor if needed
                if not isinstance(cwt_features, torch.Tensor):
                    cwt_features = torch.from_numpy(cwt_features).float()
                
                # Add batch dimension
                feats = cwt_features.unsqueeze(0)
                
                # Apply normalization
                feats = (feats - self.norm_mean) / (self.norm_std * 2)
                
                # Apply noise and augmentation if needed
                if self.noise and self.train_mode == 'train':
                    feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
                    feats = torch.roll(feats, np.random.randint(-10, 10), 1)
                
                v = {"id": idx, "source": feats}
                
                if self.is_compute_mask:
                    T = self._get_mask_indices_dims(feats.size(-1))
                    mask = compute_block_mask_1d(
                        shape=(self.clone_batch, T),
                        mask_prob=self.mask_prob,
                        mask_length=self.mask_length,
                        mask_prob_adjust=self.mask_prob_adjust,
                        inverse_mask=self.inverse_mask,
                        require_same_masks=True,
                        expand_adjcent=self.expand_adjacent,
                        mask_dropout=self.mask_dropout,
                        non_overlapping=self.non_overlapping,
                    )
                    
                    v["precomputed_mask"] = mask
                
                return v
                
            except Exception as e:
                logger.error(f"Error loading pre-computed feature {index}: {str(e)}")
                logger.warning(f"Falling back to on-the-fly computation for index {index}")
                # Fall through to on-the-fly computation
        
        # On-the-fly computation (fallback)
        try:
            # Get audio data from HuggingFace dataset
            sample = self.dataset[index]
            audio_data = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            
            # Convert to torch tensor
            audio_data = torch.tensor(audio_data).float()
            
            # Downmix to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
                audio_data = audio_data.mean(dim=0)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=16000)
                sample_rate = 16000
            
            # Normalize audio
            audio_data = audio_data - audio_data.mean()
            
            # Apply roll augmentation if needed
            if self.roll_mag_aug and self.train_mode == 'train':
                audio_data = self._roll_mag_aug(audio_data)
            
            # Compute CWT
            cwt_features = self.compute_cwt(
                audio_data,
                sample_rate=sample_rate,
                scales=64,  # Default value
                width=8.0,  # Default value
                max_len=self.target_length
            )
            
            # Convert to torch tensor
            feats = torch.from_numpy(cwt_features).float().unsqueeze(0)
            
            # Apply normalization
            feats = (feats - self.norm_mean) / (self.norm_std * 2)
            
            # Apply noise if needed
            if self.noise and self.train_mode == 'train':
                feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10
                feats = torch.roll(feats, np.random.randint(-10, 10), 1)
            
            v = {"id": idx, "source": feats}
            
            if self.is_compute_mask:
                T = self._get_mask_indices_dims(feats.size(-1))
                mask = compute_block_mask_1d(
                    shape=(self.clone_batch, T),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                    expand_adjcent=self.expand_adjacent,
                    mask_dropout=self.mask_dropout,
                    non_overlapping=self.non_overlapping,
                )
                
                v["precomputed_mask"] = mask
            
            return v
            
        except Exception as e:
            logger.error(f"Error processing index {index}: {str(e)}")
            # Return an empty placeholder if everything fails
            feats = torch.zeros((1, 64, self.target_length)).float()
            return {"id": idx, "source": feats}
    
    def __del__(self):
        # Close the HDF5 file if it's open
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass
