import numpy as np
import os
import pickle
import copy  # オブジェクトのコピーに使用
from tqdm import tqdm
import re
from tribev2.demo_utils import TribeModel, download_file
from tribev2.plotting import PlotBrain
from pathlib import Path
# os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.local/bin")

CACHE_FOLDER = Path("./cache")

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
)
plotter = PlotBrain(mesh="fsaverage5")

LOCAL_DATA_DIR = './collected_audio'
SAVE_DIR = "./tribe_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# すべてのファイルを取得
all_files = os.listdir(LOCAL_DATA_DIR)

# 正規表現パターン:
# 「(ユーザーID)_tosca_22」で始まり、その後に「_(枝番)」があってもなくても良い、最後は .wav
# 例: 5_tosca_22.wav, 5_tosca_22_72.wav 両方にマッチします
pattern = re.compile(r"^(.*)_tosca_22(_\d+)?\.wav$", re.IGNORECASE)

# ユーザーごとのファイルリストを作成
user_to_files = {}
for f in all_files:
    match = pattern.match(f)
    if match:
        user_id = match.group(1)
        if user_id not in user_to_files:
            user_to_files[user_id] = []
        user_to_files[user_id].append(os.path.join(LOCAL_DATA_DIR, f))

print(f"解析対象ユーザー数: {len(user_to_files)} 名")


# 保存ディレクトリ
SAVE_DIR_TS = "./tribe_predictions_timeseries"
os.makedirs(SAVE_DIR_TS, exist_ok=True)

for user_id, file_list in tqdm(user_to_files.items(), desc="Processing Users"):
    user_save_path = os.path.join(SAVE_DIR_TS, f"{user_id}_timeseries.npy")
    segments_save_path = os.path.join(SAVE_DIR_TS, f"{user_id}_segments.pkl")

    if os.path.exists(user_save_path) and os.path.exists(segments_save_path):
        continue

    all_sessions_preds = []
    all_sessions_segments = []
    current_time_offset = 0

    for file_path in sorted(file_list):
        try:
            df_events = model.get_events_dataframe(audio_path=file_path)
            preds, segments = model.predict(events=df_events)

            # --- Segmentオブジェクトのオフセット調整 ---
            adjusted_segments = []
            for seg in segments:
                # オブジェクトをコピーして元のデータを壊さないようにする
                new_seg = copy.copy(seg)

                # 辞書形式 [] ではなく、属性アクセス .start を使用する
                # もし .start が書き換え不可な場合は、直接代入を試みます
                try:
                    new_seg.start += current_time_offset
                except AttributeError:
                    # 万が一属性が直接書き換えられない（@propertyなど）場合の予備処理
                    pass

                adjusted_segments.append(new_seg)

            all_sessions_preds.append(preds.astype(np.float32))
            all_sessions_segments.extend(adjusted_segments)

            current_time_offset += len(preds)

        except Exception as e:
            # エラーが出た場合、どのファイルで何が起きたか詳細に出力
            print(f"\n❌ Error in {user_id} - {os.path.basename(file_path)}: {e}")

    if all_sessions_preds:
        combined_preds = np.concatenate(all_sessions_preds, axis=0)
        np.save(user_save_path, combined_preds)

        # Segmentオブジェクトのリストをそのまま保存
        with open(segments_save_path, "wb") as f:
            pickle.dump(all_sessions_segments, f)
