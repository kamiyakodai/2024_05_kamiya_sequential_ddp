import torch
import torch.autograd
import webdataset as wds
import json
import numpy as np
import io
import av
import os
import random
import tarfile

from pathlib import Path
from functools import partial
from dataset.transforms import trimmed_transform
from dataset.utils import info_from_json
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def my_collate_UCF(batch):
    return batch[0], torch.utils.data.default_collate(batch[1])


# def trimmed_video_decorder(
#     videos,
#     transform,
#     duration,
#     video_edge_time
# ):
#     for video, clip_info in videos:

#         container = av.open(io.BytesIO(video))
#         video_stream_id = 0
#         stream = container.streams.video[video_stream_id]

#         clip_len = 0
#         clip_num = 1
#         frame_idx = []
#         clip = []
#         frame_sec_list = []
#         clip_len = int(clip_info['duraion']) * int(clip_info['fps'])
#         for i, frame in enumerate(container.decode(stream)):

#             frame_idx.append(i)
#             frame_sec_list.append(frame.time)
#             img = frame.to_ndarray(format="rgb24")
#             clip.append(img)
#             if i > clip_len:  # 動画の端は読み込まない
#                 break

#             if frame.time < duration * clip_num:
#                 continue
#             else:
#                 clip = np.stack(clip, 0)  # THWC
#                 clip = np.transpose(clip, (3, 0, 1, 2))  # --> CTHW
#                 clip = torch.from_numpy(clip)
#                 subclip = transform(clip)  # transformで16フレーム均等になるように調節
#                 clip = []
#                 clip_num += 1
#                 yield subclip, clip_info
#             # ここで端を消す（残りの秒数を確認して0.5秒以下だったら抜ける）
#             if video_edge_time > (float(clip_info['duraion']) - frame.time):
#                 break


# def make_dataset(
#     shards_url,
#     dataset_size,
#     transform,
#     shuffle_buffer_size,
#     duration,
#     video_edge_time,
# ):

#     dataset = wds.WebDataset(
#         shards_url,
#         shardshuffle=True
#     )

#     dataset = dataset.decode(
#         wds.handle_extension('stats.json', lambda x: json.loads(x)),
#     )

#     dataset = dataset.to_tuple(
#         'video.bin',
#         'stats.json',
#     )

#     decode_frame = partial(
#         trimmed_video_decorder,
#         transform=transform,
#         duration=duration,
#         video_edge_time=video_edge_time,
#     )

#     dataset = dataset.compose(decode_frame)
#     dataset = dataset.shuffle(shuffle_buffer_size)
#     dataset = dataset.with_length(dataset_size)

#     return dataset


# def get_sequential_trimmed_dataset(args):
#     if args.dataset_name == 'UCF':
#         train_path = args.wds_UCF_train_path
#         val_path = args.wds_UCF_val_path
#     elif args.dataset_name == 'HMDB':
#         train_path = args.wds_HMDB_train_path
#         val_path = args.wds_HMDB_val_path

#     train_shards_path = [
#         str(path) for path in Path(train_path).glob('*.tar')
#         if not path.is_dir()
#     ]
#     val_shards_path = [
#         str(path) for path in Path(val_path).glob('*.tar')
#         if not path.is_dir()
#     ]

#     dataset_size, num_classes = info_from_json(train_path)

#     train_dataset = make_dataset(
#         shards_url=train_shards_path,
#         dataset_size=dataset_size,
#         transform=trimmed_transform(is_train=True),
#         shuffle_buffer_size=args.shuffle,
#         duration=args.clip_duration,
#         video_edge_time=args.video_edge_time,
#     )
#     val_dataset = make_dataset(
#         shards_url=val_shards_path,
#         dataset_size=dataset_size,
#         transform=trimmed_transform(is_train=False),
#         shuffle_buffer_size=args.shuffle,
#         duration=args.clip_duration,
#         video_edge_time=args.video_edge_time,
#     )

#     train_dataset = train_dataset.batched(args.batch_size)
#     val_dataset = val_dataset.batched(args.batch_size)

#     train_loader = wds.WebLoader(
#         train_dataset,
#         num_workers=args.num_workers,
#         batch_size=None,
#         collate_fn=my_collate_UCF,
#     )
#     val_loader = wds.WebLoader(
#         val_dataset,
#         num_workers=args.num_workers,
#         batch_size=None,
#         collate_fn=my_collate_UCF,
#     )

#     train_loader = train_loader.with_length(dataset_size // args.batch_size)
#     val_loader = val_loader.with_length(dataset_size // args.batch_size)

#     return train_loader, val_loader, num_classes


class UCF101Dataset(Dataset):
    def __init__(self, json_data, bin_data):
        self.data = []
        for (json_class, json_content), (bin_class, bin_content) in zip(json_data, bin_data):
            assert json_class == bin_class, "Class names do not match!"
            self.data.append((json_class, json_content, bin_content))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_name, json_content, bin_content = self.data[idx]
        # 必要に応じてjson_contentとbin_contentを処理する
        # ここでは単純にテンソルに変換して返す
        json_tensor = torch.tensor(json_content)
        bin_tensor = torch.tensor(list(bin_content))  # bin_contentはバイナリデータのまま

        return json_tensor, bin_tensor, class_name

def create_dataloader(json_data, bin_data, batch_size=32, shuffle=True, num_workers=4):
    dataset = UCF101Dataset(json_data, bin_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def open_tar(shard_filename, dir_name):
    video_list = []
    meta_list = []
    dir_list = []

    with tarfile.open(shard_filename, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                file = tar.extractfile(member)
                if file is not None:
                    if member.name.endswith('.video.bin'):
                        video_list.append(io.BytesIO(file.read()))
                    elif member.name.endswith('.stats.json'):
                        meta_list.append(json.load(file))
                    dir_list.append(dir_name.split('/')[0])

    paired_data = list(zip(video_list, meta_list))
    paired_data.sort(key=lambda x: x[0].getvalue())

    return paired_data


def make_video_dataset_list_use_shards(args, subset: str):
    if subset == 'train':
        file_txt_list = args.wds_UCF_train_path + '/'
    elif subset == 'val':
        file_txt_list = args.wds_UCF_val_path + '/'
    else:
        raise ValueError('invalid subset = ' + subset)

    json_data = []
    bin_data = []
    classes = []

    tar_files = sorted([f for f in os.listdir(file_txt_list) if f.endswith('.tar')])

    for tar_file in tar_files:
        tar_path = os.path.join(file_txt_list, tar_file)
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                file = tar.extractfile(member)
                if file is not None:
                    # ファイル名からクラス名を取得
                    class_name = os.path.dirname(member.name)
                    classes.append(class_name)

                    if member.name.endswith('.stats.json'):
                        try:
                            json_content = json.load(file)
                            json_data.append((class_name, json_content))
                        except Exception as e:
                            print(f"Error reading JSON from {member.name} in {tar_file}: {e}")
                    elif member.name.endswith('.video.bin'):
                        try:
                            bin_content = file.read()
                            bin_data.append((class_name, bin_content))
                        except Exception as e:
                            print(f"Error reading BIN from {member.name} in {tar_file}: {e}")

    return json_data, bin_data, classes


# def get_data_loader(paired_list, dir_list, batch_size, world_size, rank, shuffle):
#     # TensorDatasetの作成
#     dataset = TensorDataset(
#         torch.stack([torch.tensor(v.read()) for v, m in paired_list]),
#         torch.stack([torch.tensor(m) for v, m in paired_list])
#     )

#     # DistributedSamplerを使用してデータを各プロセスに分配
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

#     # DataLoaderを作成
#     data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

#     return data_loader

def get_num_classes(dir_list):
    unique_classes = set(dir_list)
    return len(unique_classes)

def get_train_val_loaders(args, batch_size, world_size, rank):
    # トレーニング用のpaired_listとdir_listを作成
    train_json_data, train_bin_data, n_classes = make_video_dataset_list_use_shards(args, 'train')
    # 検証用のpaired_listとdir_listを作成
    val_json_data, val_bin_data, n_classes = make_video_dataset_list_use_shards(args, 'val')

    train_loader = create_dataloader(train_json_data, train_bin_data)
    val_loader = create_dataloader(val_json_data, val_bin_data)

    return train_loader, val_loader, n_classes
