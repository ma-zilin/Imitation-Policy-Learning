"""从固定的 v2.0 提交补齐 Push-T 缓存；默认只预检，--download 才下载。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_REPO_ID = "lerobot/pusht"
DATASET_REVISION = "a16737ffb6c8dbe935291b9fd2f36d14c9bf0820"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_file(path: Path, remote) -> dict:
    """LFS 文件核对 SHA-256；普通 Git 文件核对带 blob 头的 SHA-1。"""
    size = path.stat().st_size
    if size != remote.size:
        raise ValueError(f"Size mismatch; existing file will not be overwritten: {path}")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    if remote.lfs is not None:
        matched = sha256.hexdigest() == remote.lfs.sha256
    else:
        matched = git_sha1.hexdigest() == remote.blob_id
    if not matched:
        raise ValueError(f"Checksum mismatch; existing file will not be overwritten: {path}")
    return {"size": size, "sha256": sha256.hexdigest()}


def required_names(info: dict) -> list[str]:
    """使用本地已验证的 v2.0 元数据路径，不请求最新 main 的数据布局。"""
    if info["codebase_version"] != "v2.0":
        raise ValueError("Expected local v2.0 metadata")
    names = ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/stats.json"]
    cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for episode in range(info["total_episodes"]):
        chunk = episode // info["chunks_size"]
        names.append(info["data_path"].format(episode_chunk=chunk, episode_index=episode))
        names.extend(
            info["video_path"].format(episode_chunk=chunk, episode_index=episode, video_key=camera)
            for camera in cameras
        )
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / ".cache/huggingface/lerobot" / DATASET_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/dp5_data")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    remote_info = HfApi().dataset_info(DATASET_REPO_ID, revision=DATASET_REVISION, files_metadata=True)
    if remote_info.sha != DATASET_REVISION:
        raise ValueError("Hub resolved an unexpected revision")
    remote_files = {item.rfilename: item for item in remote_info.siblings}
    names = required_names(info)
    if not set(names) <= remote_files.keys():
        raise ValueError("Pinned revision does not contain the expected files")

    existing = [name for name in names if (root / name).is_file()]
    missing = [name for name in names if not (root / name).is_file()]
    # 下载前先核对全部现有文件，拒绝静默混合版本或覆盖已有数据。
    verified = {name: verify_file(root / name, remote_files[name]) for name in existing}
    for name in names[:4]:
        if name not in verified:
            raise ValueError(f"Missing local metadata: {name}")
    download_bytes = sum(remote_files[name].size for name in missing)
    print(f"Pinned dataset: {DATASET_REPO_ID}@{DATASET_REVISION}", flush=True)
    print(f"Existing verified files={len(existing)}, missing={len(missing)}, bytes={download_bytes}", flush=True)
    if missing and not args.download:
        parser.exit(2, "Preflight only. Pass --download to fetch missing files.\n")
    if missing:
        if shutil.disk_usage(root).free < download_bytes * 2 + 1024 * 1024:
            raise ValueError("Insufficient free space for download and temporary files")
        snapshot_download(
            repo_id=DATASET_REPO_ID, repo_type="dataset", revision=DATASET_REVISION,
            local_dir=root, allow_patterns=missing, max_workers=4,
        )
    verified = {name: verify_file(root / name, remote_files[name]) for name in names}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "PASS", "dataset": DATASET_REPO_ID,
        "revision": DATASET_REVISION, "branch_at_resolution": "v2.0",
        "dataset_root": str(root), "codebase_version": info["codebase_version"],
        "file_count": len(verified), "downloaded_file_count": len(missing),
        "downloaded_bytes": download_bytes, "files": verified,
        "scope": "dataset acquisition and checksum verification only; no training",
    }
    path = output / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Pinned dataset checksum verification: PASS; {path}", flush=True)


if __name__ == "__main__":
    main()
