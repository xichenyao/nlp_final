# from torch.utils.data import Dataset
# from dataset_splits import mosei_folds, mosi_folds
# import torch
# import pickle
# import numpy as np

# class MoseiDataset(Dataset):
#     def __init__(self, data_dic, dataset="mosei", dname="train", v_max=1.0, a_max=1.0, n_samples=None):
#         self.data_dic = data_dic
#         if dataset == 'mosi':
#             fold = mosi_folds[dname]
#             self.fold_keys = [key for key in data_dic.keys() if "_".join(key.split("_")[:-1]) in fold]
#         else:
#             fold = mosei_folds[dname]
#             self.fold_keys = [key for key in data_dic.keys() if key.split('[')[0] in fold]
#         if n_samples:
#             self.fold_keys = self.fold_keys[:n_samples]
#         self.v_max = v_max
#         self.a_max = a_max

#     def __getitem__(self, idx):
#         key = self.fold_keys[idx]

#         audio = self.data_dic[key]['a']
#         audio[~np.isfinite(audio)] = 0
#         audio_normed = audio / self.a_max

#         video = self.data_dic[key]['v']
#         video[~np.isfinite(video)] = 0
#         video_normed = video / self.v_max

#         data = {"id": key, "text": self.data_dic[key]['t'], "video": video,
#                 "audio": audio, "video_normed": video_normed,
#                 "audio_normed": audio_normed, "label": self.data_dic[key]['l'][0]}
#         return data

#     def __len__(self):
#         return len(self.fold_keys)


# def sort_sequences(inputs, lengths):
#     """sort_sequences
#     Sort sequences according to lengths descendingly.

#     :param inputs (Tensor): input sequences, size [B, T, D]
#     :param lengths (Tensor): length of each sequence, size [B]
#     """
#     lengths = torch.Tensor(lengths)
#     lengths_sorted, sorted_idx = lengths.sort(descending=True)
#     _, unsorted_idx = sorted_idx.sort()
#     return inputs[sorted_idx], lengths_sorted, sorted_idx


# def collate_fn(batch):
#     MAX_LEN = 128

#     lens = [min(len(row["text"]), MAX_LEN) for row in batch]

#     tdims = batch[0]["text"].shape[1]
#     adims = batch[0]["audio"].shape[1]
#     vdims = batch[0]["video"].shape[1]

#     bsz, max_seq_len = len(batch), max(lens)

#     text_tensor = torch.zeros((bsz, max_seq_len, tdims))

#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         text_tensor[i_batch, :length] = torch.Tensor(input_row["text"][:length])

#     video_tensor = torch.zeros((bsz, max_seq_len, vdims))
#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         video_tensor[i_batch, :length] = torch.Tensor(input_row["video"][:length])


#     audio_tensor = torch.zeros((bsz, max_seq_len, adims))
#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         audio_tensor[i_batch, :length] = torch.Tensor(input_row["audio"][:length])

#     video_normed_tensor = torch.zeros((bsz, max_seq_len, vdims))
#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         video_normed_tensor[i_batch, :length] = torch.Tensor(input_row["video_normed"][:length])

#     audio_normed_tensor = torch.zeros((bsz, max_seq_len, adims))
#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         audio_normed_tensor[i_batch, :length] = torch.Tensor(input_row["audio_normed"][:length])

#     tgt_tensor = torch.stack([torch.tensor(row["label"]) for row in batch])

#     text_tensor, lens, sorted_idx = sort_sequences(text_tensor, lens)

#     return text_tensor, video_tensor[sorted_idx], audio_tensor[sorted_idx], \
#            video_normed_tensor[sorted_idx], audio_normed_tensor[sorted_idx], tgt_tensor[sorted_idx], lens


# if __name__ == "__main__":

#     with open("data/MOSEI/mosei.dataset", "rb") as f:
#         data_dic = pickle.load(f)

#     train_dataset = MoseiDataset(data_dic, "train")

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

#     for text_tensor, video_tensor, audio_tensor, tgt_tensor, video_normed_tensor, audio_normed_tensor, lens in train_loader:
#         print(text_tensor.size(), video_tensor.size(), audio_tensor.size(),
#               video_normed_tensor.size(), audio_normed_tensor.size(), tgt_tensor.size(), lens.size())

#         exit()
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from dataset_splits import mosei_folds, mosi_folds

class MoseiDataset(Dataset):
    def __init__(self, data_dic, dataset="mosei", dname="train", v_max=1.0, a_max=1.0, n_samples=None):
        self.data_dic = data_dic
        self.v_max = v_max
        self.a_max = a_max

        # 选择折叠文件
        fold = mosei_folds[dname] if dataset == "mosei" else mosi_folds[dname]
        self.fold_keys = [
            key for key in data_dic.keys() if (key.split('[')[0] if dataset == "mosei" else "_".join(key.split("_")[:-1])) in fold
        ]
        
        # 如果指定了样本数量，进行限制
        if n_samples:
            self.fold_keys = self.fold_keys[:n_samples]

    def __getitem__(self, idx):
        key = self.fold_keys[idx]
        sample = self.data_dic[key]

        # 音频归一化并处理无效值
        audio = np.nan_to_num(sample['a'], nan=0.0)
        audio_normed = audio / self.a_max

        # 视频归一化并处理无效值
        video = np.nan_to_num(sample['v'], nan=0.0)
        video_normed = video / self.v_max

        # 返回数据
        return {
            "id": key,
            "text": sample['t'],
            "audio": audio,
            "video": video,
            "audio_normed": audio_normed,
            "video_normed": video_normed,
            "label": sample['l'][0]  # 假设标签是一个包含单一元素的列表
        }

    def __len__(self):
        return len(self.fold_keys)


def sort_sequences(inputs, lengths):
    """Sort sequences according to lengths."""
    lengths = torch.tensor(lengths)
    sorted_len, sorted_idx = lengths.sort(descending=True)
    return inputs[sorted_idx], sorted_len, sorted_idx


def collate_fn(batch):
    MAX_LEN = 128
    bsz = len(batch)

    # 计算每个样本的文本长度，并限制最大长度
    lens = [min(len(row["text"]), MAX_LEN) for row in batch]
    tdims, adims, vdims = batch[0]["text"].shape[1], batch[0]["audio"].shape[1], batch[0]["video"].shape[1]

    max_len = max(lens)
    text_tensor = torch.zeros((bsz, max_len, tdims), dtype=torch.float32)
    video_tensor = torch.zeros((bsz, max_len, vdims), dtype=torch.float32)
    audio_tensor = torch.zeros((bsz, max_len, adims), dtype=torch.float32)
    video_normed_tensor = torch.zeros((bsz, max_len, vdims), dtype=torch.float32)
    audio_normed_tensor = torch.zeros((bsz, max_len, adims), dtype=torch.float32)

    # 填充每个 batch 的数据
    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = torch.tensor(input_row["text"][:length], dtype=torch.float32)
        video_tensor[i_batch, :length] = torch.tensor(input_row["video"][:length], dtype=torch.float32)
        audio_tensor[i_batch, :length] = torch.tensor(input_row["audio"][:length], dtype=torch.float32)
        video_normed_tensor[i_batch, :length] = torch.tensor(input_row["video_normed"][:length], dtype=torch.float32)
        audio_normed_tensor[i_batch, :length] = torch.tensor(input_row["audio_normed"][:length], dtype=torch.float32)

    # 目标标签
    tgt_tensor = torch.stack([torch.tensor(row["label"], dtype=torch.float32) for row in batch])

    # 排序
    text_tensor, lens, sorted_idx = sort_sequences(text_tensor, lens)
    return text_tensor, video_tensor[sorted_idx], audio_tensor[sorted_idx], \
           video_normed_tensor[sorted_idx], audio_normed_tensor[sorted_idx], tgt_tensor[sorted_idx], lens


if __name__ == "__main__":
    # 加载数据集
    with open("data/MOSEI/mosei.dataset", "rb") as f:
        data_dic = pickle.load(f)

    train_dataset = MoseiDataset(data_dic, "train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )

    for text_tensor, video_tensor, audio_tensor, tgt_tensor, video_normed_tensor, audio_normed_tensor, lens in train_loader:
        print(f"text_tensor: {text_tensor.size()}")
        print(f"video_tensor: {video_tensor.size()}")
        print(f"audio_tensor: {audio_tensor.size()}")
        print(f"video_normed_tensor: {video_normed_tensor.size()}")
        print(f"audio_normed_tensor: {audio_normed_tensor.size()}")
        print(f"tgt_tensor: {tgt_tensor.size()}")
        print(f"lens: {lens.size()}")
        break  # 只打印一个 batch

# 1. 减少多余的计算
# 目前在 __getitem__ 和 collate_fn 中，数据的归一化和填充操作每次都进行，可以考虑将某些操作移到初始化时，或者避免重复计算。在 __getitem__ 中对音频和视频的归一化操作，每次都做一次，而不是在数据加载阶段做。
# 使用 np.nan_to_num 来代替手动设置 ~np.isfinite，使得音频和视频数据的处理更简洁。
# 通过字典提取数据并统一处理，减少了每次访问数据时的计算开销。
# 2. 减少内存使用
# 在 collate_fn 中，创建了很多大的 Tensor（如 text_tensor, video_tensor, audio_tensor）。考虑只在需要时才创建并进行填充，减少内存消耗，尤其是在处理长序列时。
# 采用 torch.zeros 而不是在每次循环中创建 Tensor，这样避免了多次重新分配内存。
# 3. 优化 sort_sequences
# 在排序过程中，使用 torch.Tensor 来存储长度，并进行排序。可以优化成只一次排序，减少计算量。
# 直接返回排序后的数据
# 4. 数据归一化改进
# 在 __getitem__ 中，归一化操作可以通过使用 np.nan_to_num 或者 torch 来更简洁地处理。


