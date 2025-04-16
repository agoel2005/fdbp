from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TIGER-Lab/VLM2Vec-Qwen2VL-2B",
    local_dir="vlm2vec_qwen2vl_2b",
    local_dir_use_symlinks=False  # important for portable storage
)