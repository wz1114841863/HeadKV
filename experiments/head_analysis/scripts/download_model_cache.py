import os
import argparse
import fnmatch
import gc
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ.pop("HF_HUB_DISABLE_XET", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


# ============================================================
# 模型列表
# ============================================================

MODELS = [
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
]


# 为需要严格固定版本的模型指定 commit.
# 其他模型默认使用 main.
MODEL_REVISIONS = {
    "NousResearch/Meta-Llama-3.1-8B": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
}


# ============================================================
# 文件筛选规则
# ============================================================

# 允许下载的文件.
ALLOW_PATTERNS = [
    # Safetensors 权重
    "*.safetensors",
    "*.safetensors.index.json",
    # 模型/生成和处理器配置
    "*.json",
    # Tokenizer 文件
    "tokenizer*",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.txt",
    "vocab.json",
    # 自定义模型代码和模板
    "*.py",
    "*.jinja",
    # 说明和许可证
    "README*",
    "LICENSE*",
    "NOTICE*",
    "USE_POLICY*",
    ".gitattributes",
]

# 强制排除的文件.
# ignore 规则的优先级高于 allow 规则.
IGNORE_PATTERNS = [
    # Meta 原始格式目录
    "original/*",
    "original/**",
    # 非 Safetensors 权重
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    # 其他框架格式
    "*.flax",
    "*.h5",
    "*.tflite",
    "*.msgpack",
    # 量化或其他推理格式
    "*.gguf",
    "*.ggml",
    "*.onnx",
    # 训练状态
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
]


# ============================================================
# 基础工具
# ============================================================


def matches_any(filename: str, patterns: list[str]) -> bool:
    """判断仓库文件路径是否匹配任意模式."""
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def should_download(filename: str) -> bool:
    """
    判断文件是否应该下载.

    ignore 优先:
    如果同时匹配 allow 和 ignore,则不下载.
    """
    if matches_any(filename, IGNORE_PATTERNS):
        return False

    return matches_any(filename, ALLOW_PATTERNS)


def clear_memory() -> None:
    """释放 Python 和 CUDA 缓存."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def get_load_dtype() -> torch.dtype:
    """根据设备能力选择模型加载精度."""
    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def verify_snapshot(snapshot_path: str) -> None:
    """检查本地快照是否包含基本配置和 Safetensors 权重."""
    root = Path(snapshot_path)

    if not root.exists():
        raise FileNotFoundError(f"快照目录不存在:{snapshot_path}")

    config_path = root / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"快照中缺少 config.json:{snapshot_path}")

    weight_files = sorted(root.rglob("*.safetensors"))

    if not weight_files:
        raise FileNotFoundError(f"快照中没有发现 Safetensors 权重:{snapshot_path}")

    total_size = sum(path.stat().st_size for path in weight_files)

    print(f"[Verify] Safetensors 文件数量:{len(weight_files)}")
    print(f"[Verify] Safetensors 总大小:{total_size / 1024**3:.2f} GiB")


# ============================================================
# 方案一:snapshot_download
# ============================================================


def download_with_snapshot(
    model_name: str,
    revision: str,
    max_workers: int,
) -> str:
    """
    使用 snapshot_download 下载经过筛选的模型快照.
    """
    return snapshot_download(
        repo_id=model_name,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=max_workers,
    )


# ============================================================
# 方案二:逐文件下载
# ============================================================


def find_snapshot_root(file_path: str, commit_hash: str) -> Path:
    """
    根据 hf_hub_download 返回的文件路径找到 snapshot 根目录.

    文件路径通常类似:
        .../snapshots/<commit_hash>/config.json
    """
    path = Path(file_path)

    candidates = [path.parent, *path.parents]

    for candidate in candidates:
        if candidate.name == commit_hash:
            return candidate

    raise RuntimeError(f"无法从下载路径定位快照目录:{file_path}")


def download_single_file_with_retry(
    model_name: str,
    filename: str,
    revision: str,
    max_retries: int,
) -> str:
    """逐文件下载并重试."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[File] {filename} " f"({attempt}/{max_retries})")

            return hf_hub_download(
                repo_id=model_name,
                filename=filename,
                revision=revision,
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print(f"[File warning] {filename}: " f"{type(exc).__name__}: {exc}")

            if attempt >= max_retries:
                raise

            wait_seconds = min(attempt * 5, 30)
            print(f"[File retry] {wait_seconds} 秒后重试")
            time.sleep(wait_seconds)

    raise RuntimeError(f"文件下载失败:{filename}")


def download_sequentially(
    model_name: str,
    revision: str,
    max_retries: int,
) -> str:
    """
    获取仓库文件列表并逐个下载.

    优点:
    1. 不会同时发起多个下载;
    2. 可以直接看到具体是哪个文件失败;
    3. 可以严格排除 original/ 和 .pth;
    4. 已经缓存的文件不会重复下载.
    """
    print("[Fallback] 改为逐文件下载")

    api = HfApi(endpoint="https://huggingface.co")

    info = api.model_info(
        repo_id=model_name,
        revision=revision,
    )

    commit_hash = info.sha

    if not commit_hash:
        raise RuntimeError(f"无法获取模型 commit hash:{model_name}")

    repo_files = [sibling.rfilename for sibling in info.siblings if sibling.rfilename]

    selected_files = [filename for filename in repo_files if should_download(filename)]

    # 先下载小型配置文件,再下载大型 Safetensors.
    selected_files.sort(
        key=lambda name: (
            name.endswith(".safetensors"),
            name,
        )
    )

    print(f"[Repo] 仓库文件数量:{len(repo_files)}")
    print(f"[Repo] 筛选后文件数量:{len(selected_files)}")
    print(f"[Commit] {commit_hash}")

    if not selected_files:
        raise RuntimeError(f"筛选后没有需要下载的文件:{model_name}")

    downloaded_paths: list[str] = []

    for filename in selected_files:
        local_path = download_single_file_with_retry(
            model_name=model_name,
            filename=filename,
            revision=commit_hash,
            max_retries=max_retries,
        )

        downloaded_paths.append(local_path)

    snapshot_root = find_snapshot_root(
        file_path=downloaded_paths[0],
        commit_hash=commit_hash,
    )

    return str(snapshot_root)


# ============================================================
# 总下载函数
# ============================================================


def robust_download(
    model_name: str,
    max_retries: int = 5,
    max_workers: int = 1,
) -> Optional[str]:
    """
    首先使用 snapshot_download.

    如果失败,则自动切换为逐文件下载,以定位并绕过并发问题.
    """
    revision = MODEL_REVISIONS.get(model_name, "main")

    print(f"\n{'=' * 80}")
    print(f"[Download] 模型:{model_name}")
    print(f"[Revision] {revision}")
    print(f"[Endpoint] {os.getenv('HF_ENDPOINT')}")
    print(f"[huggingface_hub] {huggingface_hub.__version__}")
    print(f"[Max workers] {max_workers}")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Snapshot] 第 {attempt}/{max_retries} 次尝试")

            snapshot_path = download_with_snapshot(
                model_name=model_name,
                revision=revision,
                max_workers=max_workers,
            )

            verify_snapshot(snapshot_path)

            print(f"[Download] ✅ 下载完成:{model_name}")
            print(f"[Snapshot] {snapshot_path}")

            return snapshot_path

        except KeyboardInterrupt:
            print("\n[Interrupted] 用户终止下载.")
            raise

        except Exception as exc:
            print(f"[Snapshot warning] " f"{type(exc).__name__}: {exc}")

            if attempt < max_retries:
                wait_seconds = min(attempt * 5, 30)
                print(f"[Snapshot retry] {wait_seconds} 秒后重试")
                time.sleep(wait_seconds)

    # snapshot_download 多次失败后,逐文件下载.
    try:
        snapshot_path = download_sequentially(
            model_name=model_name,
            revision=revision,
            max_retries=max_retries,
        )

        verify_snapshot(snapshot_path)

        print(f"[Download] ✅ 逐文件下载完成:{model_name}")
        print(f"[Snapshot] {snapshot_path}")

        return snapshot_path

    except KeyboardInterrupt:
        raise

    except Exception as exc:
        print(f"[Error] 模型下载失败:{model_name}\n" f"[Exception] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


# ============================================================
# 模型加载校验
# ============================================================


def touch_model(
    model_name: str,
    snapshot_path: str,
    trust_remote_code: bool = True,
) -> bool:
    """从本地快照加载并校验模型."""
    print(f"\n[Loading] {model_name}")
    print(f"[Local snapshot] {snapshot_path}")

    dtype = get_load_dtype()

    print(f"[DType] {dtype}")
    print(f"[CUDA available] {torch.cuda.is_available()}")

    try:
        config = AutoConfig.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

        load_kwargs = {
            "config": config,
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            **load_kwargs,
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        print(f"[Config] model_type={config.model_type}")
        print(f"[Tokenizer] vocab_size={len(tokenizer):,}")
        print(f"[Parameters] {parameter_count:,}")
        print(f"[OK] 成功加载并校验:{model_name} " f"(Safetensors, {dtype})")

        del model
        del tokenizer
        del config

        clear_memory()
        return True

    except KeyboardInterrupt:
        clear_memory()
        raise

    except Exception as exc:
        print(f"[Error] 加载失败:{model_name}\n" f"[Exception] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        clear_memory()
        return False


# ============================================================
# 参数解析
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并校验 Hugging Face Safetensors 模型")

    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=("指定模型,可以重复使用." "未提供时下载 MODELS 中的全部模型."),
    )

    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="只下载模型,不加载完整权重.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="每种下载方式的最大重试次数,默认 5.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="snapshot_download 并发数,默认 1.",
    )

    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="禁用 trust_remote_code.",
    )

    return parser.parse_args()


# ============================================================
# 主程序
# ============================================================


def main() -> int:
    args = parse_args()

    if args.max_retries < 1:
        raise ValueError("--max-retries 必须大于或等于 1")

    if args.max_workers < 1:
        raise ValueError("--max-workers 必须大于或等于 1")

    models = args.models or MODELS
    trust_remote_code = not args.no_trust_remote_code

    print("=" * 80)
    print("Hugging Face 模型下载与校验")
    print("=" * 80)
    print(f"HF_HOME={os.getenv('HF_HOME')}")
    print(f"HF_ENDPOINT={os.getenv('HF_ENDPOINT')}")
    print("HF_HUB_ETAG_TIMEOUT=" f"{os.getenv('HF_HUB_ETAG_TIMEOUT')}")
    print("HF_HUB_DOWNLOAD_TIMEOUT=" f"{os.getenv('HF_HUB_DOWNLOAD_TIMEOUT')}")
    print("huggingface_hub=" f"{huggingface_hub.__version__}")
    print(f"待处理模型数量={len(models)}")

    failed_models: list[str] = []

    for model_name in models:
        snapshot_path = robust_download(
            model_name=model_name,
            max_retries=args.max_retries,
            max_workers=args.max_workers,
        )

        if snapshot_path is None:
            failed_models.append(model_name)
            continue

        if args.skip_load_check:
            continue

        loaded = touch_model(
            model_name=model_name,
            snapshot_path=snapshot_path,
            trust_remote_code=trust_remote_code,
        )

        if not loaded:
            failed_models.append(model_name)

    print(f"\n{'=' * 80}")
    print("[Summary]")

    if failed_models:
        print("以下模型处理失败:")

        for model_name in failed_models:
            print(f"  - {model_name}")

        return 1

    print("✅ 所有模型处理完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
