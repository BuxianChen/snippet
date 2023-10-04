import requests
import tempfile
from contextlib import contextmanager
from functools import partial
import os
from tqdm import tqdm
import shutil


def hf_hub_download(
    repo_id,
    filename,
    revision,
    endpoint,
    repo_type,
    token,
    cache_dir,
    resume_download: bool,
):
    url = f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"
    headers = {
        "user-agent": "mylib/v1.0; hf_hub/0.17.2; python/3.9.16; torch/1.12.1+cu113;",
        "authorization": f"Bearer {token}"
    }
    meta_headers = headers.copy()
    meta_headers["Accept-Encoding"] = "identity"

    meta_resp = requests.head(url, headers=meta_headers)
    
    # lfs 文件是 X-Linked-ETag, X-Linked-Size, Location
    etag = meta_resp.headers.get("X-Linked-ETag") or meta_resp.headers.get("ETag")
    size = meta_resp.headers.get("X-Linked-Size") or meta_resp.headers.get("Content-Length")
    location = meta_resp.headers.get("Location") or meta_resp.request.url
    metadata = {
        "commit_hash": meta_resp.headers["X-Repo-Commit"],
        "etag": etag,
        "size": size,
        "location": location,
    }

    expected_size = int(metadata["size"])
    url_to_download = metadata["location"]
    commit_hash = metadata["commit_hash"]
    etag = metadata["etag"].replace('"', '')  # 待研究是否与 git 的 blob-id 的计算方式相同

    storage_folder = os.path.join(cache_dir, "--".join([f"{repo_type}s", *repo_id.split("/")]))
    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = os.path.join(storage_folder, "snapshots", revision, filename)
    ref_path = os.path.join(storage_folder, "refs", revision)

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)

    if revision != commit_hash:
        with open(ref_path, "w") as fw:
            fw.write(commit_hash)

    if resume_download:
        incomplete_path = blob_path + ".incomplete"
        @contextmanager
        def _resumable_file_manager():
            with open(incomplete_path, "ab") as f:
                yield f
        temp_file_manager = _resumable_file_manager
        if os.path.exists(incomplete_path):
            resume_size = os.stat(incomplete_path).st_size
        else:
            resume_size = 0
    else:
        temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
        resume_size = 0

    if url_to_download != url:
        headers.pop("authorization", None)

    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size}-"

    # 原始实现此处还用了 filelock
    with temp_file_manager() as temp_file:
        r = requests.get(
            url_to_download,
            stream=True,
            headers=headers
        )
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            total = resume_size + int(content_length)
        else:
            total = None

        progress = tqdm(total=total, initial=resume_size)
        for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)
        
        if expected_size != temp_file.tell():
            raise
        progress.close()

    # _chmod_and_replace(temp_file.name, blob_path)
    shutil.move(temp_file.name, blob_path)

    src = blob_path
    dst = pointer_path
    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))
    relative_src = os.path.relpath(abs_src, os.path.dirname(abs_dst))
    os.symlink(relative_src, abs_dst)
    return pointer_path

if __name__ == "__main__":
    os.environ['HTTP_PROXY'] = "http://172.18.48.1:7890"
    os.environ['HTTPS_PROXY'] = "http://172.18.48.1:7890"
    
    endpoint = "https://huggingface.co"
    repo_type = "model"
    # 可以使用如下命令创建一个大小为 50M 的文件, 提前用 upload_file 上传
    # fallocate -l 52428800 test.bin
    filename = ".gitattributes"  # "test.bin"
    revision = "main"
    repo_id = "Buxian/test-model"
    resume_download = True
    cache_dir = "./hf_cache_test"
    token = "hf_xyzz"
    pointer_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        endpoint=endpoint,
        token=token,
        repo_type=repo_type,
        cache_dir=cache_dir,
        resume_download=resume_download
    )
    
    # 以下验证仅使用于非 lfs 文件
    blob_path = os.path.realpath(pointer_path)
    print("pointer_path:", pointer_path)
    print("blob_path:", blob_path)
    import hashlib
    import os
    blob_id = os.path.basename(blob_path)  # a6344aac8c09253b3b630fb776ae94478aa0275b
    filename = pointer_path                # f"hf_cache_test/models--Buxian--test-model/blobs/{blob_id}"
    # 这里的计算方法即为 git 计算 blob id 的算法
    size = os.stat(filename).st_size
    prefix = f"blob {size}\0".encode()
    with open(filename, "rb") as fr:
        content = fr.read()
    check_blob_id = hashlib.sha1(prefix + content).hexdigest()
    assert check_blob_id == blob_id
