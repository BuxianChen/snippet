from huggingface_hub._commit_api import CommitOperationAdd, CommitOperationCopy
from huggingface_hub.utils._http import get_session
from huggingface_hub import HfApi
from typing import List, Dict
import itertools
import base64
import os
import json

os.environ['HTTP_PROXY'] = "http://172.18.48.1:7890"
os.environ['HTTPS_PROXY'] = "http://172.18.48.1:7890"

endpoint = "https://huggingface.co"

token = "hf_xxxxx"  # modify this!!!
repo_id = "Buxian/test-model"
repo_type = "model"
revision = "main"
create_pr = False
commit_message = "commit title"
commit_description = "commit details"

path_or_fileobj = "xyz.txt"
path_in_repo = "test/sample.txt"

# upload_file 仅包含一个文件
ops = [
    CommitOperationAdd(
        path_in_repo = path_in_repo,
        path_or_fileobj=path_or_fileobj
    )
]


# ======== 各类操作的划分 =========
# copies = [
#     CommitOperationCopy(
#         src_path_in_repo="test/sample.txt",
#         src_revision="main",
#         path_in_repo="test/sample_from_main.txt",
#     ),
#     CommitOperationCopy(
#         src_path_in_repo="test/bin_sample.bin",
#         src_revision="main",
#         path_in_repo="test/sample_from_main_copy.bin",
#     ),
# ]

# additions = [
#     CommitOperationAdd(
#         path_or_fileobj = "xyz.txt",
#         path_in_repo="bin/xyz.bin",
#     ),
#     CommitOperationAdd(
#         path_or_fileobj = "xyz.txt",
#         path_in_repo="bin/xyz.txt",
#     ),
# ]
# ops = copies + additions

# additions: lfs/non-lfs
# copies: lfs
# deletes: lfs/non-lfs


# ==================== create_commit START ==================


additions = [op for op in ops if isinstance(op, CommitOperationAdd)]
copies = [op for op in ops if isinstance(op, CommitOperationCopy)]

# ======== step 1: fetch_lfs_files_to_copy: 确定additions中每个文件的上传方式是 regular 还是 lfs ========

headers = {
  "user-agent": "mylib/v1.0; hf_hub/0.17.2; python/3.9.16; torch/1.12.1+cu113;",
  "authorization": f"Bearer {token}",
}

payload = {
    "files": [
        {
            "path": op.path_in_repo,
            "sample": base64.b64encode(op.upload_info.sample).decode("ascii"),
            "size": op.upload_info.size,
            "sha": op.upload_info.sha256.hex(),
        }
        for op in additions
    ]
}

url = f"{endpoint}/api/{repo_type}s/{repo_id}/preupload/{revision}"
preupload_info = get_session().post(
    url,
    json=payload,
    headers=headers,
    params={"create_pr": "1"} if create_pr else None
).json()

# {'files': [{'path': 'test/sample.txt', 'uploadMode': 'regular'}]}
print(preupload_info)

# file["uploadModel"] 的取值只有 "lfs" 和 "regular"
upload_modes = {
    file["path"]: file["uploadMode"] for file in preupload_info["files"]
}

for op in additions:
    if op.upload_info.size == 0:
        path = op.path_in_repo
        upload_modes[path] = "regular"

# ============== step 2: fetch_lfs_files_to_copy: 查询copies中可以进行lfs拷贝所需信息(copies中都是lfs文件) ========

# CommitOperationCopy 用于提交将某个分支某个文件拷贝进行提交
# src_path_in_repo="asset/model.bin"
# path_in_repo="asset/new_model.bin"
# src_revision="dev"
# 将dev分支的asset/model.bin拷贝到main分支的asset/new_model.bin

# @dataclass
# class CommitOperationCopy:
#     src_path_in_repo: str
#     path_in_repo: str
#     src_revision: Optional[str] = None

# @dataclass
# class CommitOperationAdd:
#     path_in_repo: str
#     path_or_fileobj: Union[str, Path, bytes, BinaryIO]
#     upload_info: UploadInfo = field(init=False, repr=False)

files_to_copy = {}
hf_api = HfApi(endpoint=endpoint, token=token)
for src_revision, operations in itertools.groupby(copies, key=lambda op: op.src_revision):
    paths = [op.src_path_in_repo for op in operations]
    src_repo_files = hf_api.list_files_info(
        repo_id=repo_id,
        paths=paths,
        revision=src_revision or revision,
        repo_type=repo_type,
    )
    for src_repo_file in src_repo_files:
        # 非lfs文件'lfs'的取值为None
        # RepoFile: { 
        #     {'blob_id': '600c48de5e368ec6822ebddfccab8f3405913ebc',
        #     'lastCommit': {'date': '2023-09-27T09:49:04.000Z',
        #                     'id': 'f9ec1fee0dec37acd83a61a8e975175b41a20149',
        #                     'title': 'Upload test/bin_sample.bin with huggingface_hub'},
        #     'lfs': {'pointer_size': 127,
        #             'sha256': '4252f8d56b4bb236d0b1bc95a1202e392ca84ce0644bf628398fbb9517287da8',
        #             'size': 12},
        #     'rfilename': 'test/bin_sample.bin',
        #     'security': {'avScan': {'virusFound': False, 'virusNames': None},
        #                     'blobId': '600c48de5e368ec6822ebddfccab8f3405913ebc',
        #                     'name': 'test/bin_sample.bin',
        #                     'pickleImportScan': None,
        #                     'safe': True},
        #     'size': 12}
        #     }
        assert src_repo_file.lfs, "Copying a non-LFS file is not implemented"
        files_to_copy[(src_repo_file.rfilename, src_revision)] = src_repo_file

    for operation in operations:
        if (operation.src_path_in_repo, src_revision) not in files_to_copy:
            raise ValueError(f"Cannot copy {operation.src_path_in_repo} at revision {src_revision or revision}: file is missing on repo.")

# ============== step 3: upload_lfs_files (实际执行 additions 中的lfs文件上传)========

lfs_additions = [op for op in additions if upload_modes[op.path_in_repo]=="lfs"]

batch_actions: List[Dict] = []

# Learn more: https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md
url = f"{endpoint}/{repo_id}.git/info/lfs/objects/batch"
LFS_HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}

for i in range(0, len(lfs_additions), 256):
    batched_lfs_additions = lfs_additions[i:i+256]
    from requests.auth import HTTPBasicAuth
    response_json = get_session().post(
        url,
        headers=LFS_HEADERS,
        auth=HTTPBasicAuth("access_token", token),
        json={
            "operation": "upload",
            "transfers": ["basic", "multipart"],
            "objects": [
                {
                    "oid": upload.upload_info.sha256.hex(),
                    "size": upload.upload_info.size,
                }
                for upload in batched_lfs_additions
            ],
            "hash_algo": "sha256",
        },
    ).json()
    objects = response_json['objects']
    # response_json
    # {
    #     'transfer': 'basic',
    #     'objects': [
    #         {'oid': '4252f8d56b4bb236d0b1bc95a1202e392ca84ce0644bf628398fbb9517287da8', 'size': 12}
    #     ]
    # }
    assert all(["error" not in obj for obj in objects])
    batch_actions += objects

oid2addop = {add_op.upload_info.sha256.hex(): add_op for add_op in lfs_additions}
batch_actions = [a for a in batch_actions if a.get("actions") is not None]


# 此处可以并行实现
for batch_action in batch_actions:
    operation = oid2addop[batch_action["oid"]]
    # 待研究
    lfs_upload(operation=operation, lfs_batch_action=batch_action, token=token)

# ============== step 4: prepare_commit_payload (实际执行 additions 中非lfs文件上传, copies 中 lfs 拷贝, delete文件删除) ========
# 待研究
commit_payload = prepare_commit_payload(
    operations=ops,               # 来自于 step 1
    upload_modes=upload_modes,    # 来自于 step 1
    files_to_copy=files_to_copy,  # 来自于 step 2
)

# ============== step 5: 创建 commit 请求 ===============
commit_url = f"{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"
headers = {
    "Content-Type": "application/x-ndjson",
    **headers,  # step 1 中的 headers
}
def _payload_as_ndjson():
    for item in commit_payload:
        yield json.dumps(item).encode()
        yield b"\n"
data = b"".join(_payload_as_ndjson())

commit_resp = get_session().post(
    url=commit_url,
    headers=headers,
    data=data,
    params={"create_pr": "1"} if create_pr else None
)
commit_data = commit_resp.json()

commit_info = CommitInfo(
    commit_url=commit_data["commitUrl"],
    commit_message=commit_message,
    commit_description=commit_description,
    oid=commit_data["commitOid"],
    pr_url=commit_data["pullRequestUrl"] if create_pr else None,
)

# ==================== create_commit END ==================

# 返回值:
# Similar to `hf_hub_url` but it's "blob" instead of "resolve"
path = f"{endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"


def lfs_upload(operation: "CommitOperationAdd", lfs_batch_action: Dict, token):
    actions = lfs_batch_action.get("actions")
    upload_action = lfs_batch_action["actions"]["upload"]
    verify_action = lfs_batch_action["actions"].get("verify")
    header = upload_action.get("header", {})
    chunk_size = header.get("chunk_size")
    if chunk_size is not None:
        _upload_multi_part(operation=operation, header=header, chunk_size=chunk_size, upload_url=upload_action["href"])
    else:
        _upload_single_part(operation=operation, upload_url=upload_action["href"])
    if verify_action is not None:
        verify_resp = get_session().post(
            verify_action["href"],
            auth=HTTPBasicAuth(username="USER", password=token),  # type: ignore
            json={"oid": operation.upload_info.sha256.hex(), "size": operation.upload_info.size},
        )

def prepare_commit_payload():
    pass
