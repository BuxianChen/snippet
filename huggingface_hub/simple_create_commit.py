# from huggingface_hub._commit_api import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.hf_api import CommitInfo
from huggingface_hub.lfs import SliceFileObj
from huggingface_hub.utils._http import get_session
from huggingface_hub import HfApi
from typing import List, Dict
import itertools
import base64
import os
import json


# CommitOperationCopy 用于提交将某个分支某个文件拷贝进行提交
# src_path_in_repo="asset/model.bin"
# path_in_repo="asset/new_model.bin"
# src_revision="dev"
# 将dev分支的asset/model.bin拷贝到main分支的asset/new_model.bin
@dataclass
class CommitOperationCopy:
    src_path_in_repo: str
    path_in_repo: str
    src_revision: Optional[str] = None

class CommitOperationDelete:
    path_in_repo: str
    is_folder: Union[bool, Literal["auto"]] = "auto"

@dataclass
class CommitOperationAdd:
    path_in_repo: str
    path_or_fileobj: Union[str, Path, bytes, BinaryIO]
    upload_info: UploadInfo = field(init=False, repr=False)
    @contextmanager
    def as_file(self, with_tqdm: bool = False) -> Iterator[BinaryIO]:
        if isinstance(self.path_or_fileobj, str) or isinstance(self.path_or_fileobj, Path):
            if with_tqdm:
                with tqdm_stream_file(self.path_or_fileobj) as file:
                    yield file
            else:
                with open(self.path_or_fileobj, "rb") as file:
                    yield file
        elif isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
            prev_pos = self.path_or_fileobj.tell()
            yield self.path_or_fileobj
            self.path_or_fileobj.seek(prev_pos, io.SEEK_SET)
    def b64content(self) -> bytes:
        with self.as_file() as file:
            return base64.b64encode(file.read())


def create_commit(
    endpoint, repo_id, repo_type, revision, token, create_pr,
    commit_message, commit_description, parent_commit,
    operations
    ):
    additions = [op for op in operations if isinstance(op, CommitOperationAdd)]
    copies = [op for op in operations if isinstance(op, CommitOperationCopy)]
    # upload_modes: op.upload_info.path_in_repo -> "regular/lfs"
    upload_modes = fetch_upload_modes(endpoint, repo_id, repo_type, revision, token, create_pr, additions)
    # files_to_copy: (op.path_in_repo, op.src_revision) -> src_repo_file(RepoFile)
    files_to_copy = fetch_lfs_files_to_copy(endpoint, repo_id, repo_type, revision, token, create_pr, copies)
    lfs_additions = [op for op in additions if upload_modes[op.path_in_repo]=="lfs"]
    upload_lfs_files(endpoint, repo_id, repo_type, revision, token, create_pr, lfs_additions)
    # commit_payload: 一个yield字节的迭代器
    commit_payload = prepare_commit_payload(operations, upload_modes, files_to_copy, commit_message, commit_description, parent_commit)
    commit_info = do_commit(endpoint, repo_id, repo_type, revision, token, create_pr, commit_payload)
    return commit_info
    

# ======== step 1: fetch_upload_modes: 确定additions中每个文件的上传方式是 regular 还是 lfs ========
def fetch_upload_modes(endpoint, repo_id, repo_type, revision, token, create_pr, additions):
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
        params={"create_pr": "1"} if create_pr else None  # 这个参数重要吗?
    ).json()
    # preupload_info: {'files': [{'path': 'test/sample.txt', 'uploadMode': 'regular'}]}
    # uploadMode 的取值只有 "lfs" 和 "regular"
    upload_modes = {
        file["path"]: file["uploadMode"] for file in preupload_info["files"]
    }

    for op in additions:
        if op.upload_info.size == 0:
            path = op.path_in_repo
            upload_modes[path] = "regular"
    return upload_modes

# ============== step 2: fetch_lfs_files_to_copy: 查询copies中可以进行lfs拷贝所需信息(copies中都是lfs文件) ========
def fetch_lfs_files_to_copy(endpoint, repo_id, repo_type, revision, token, create_pr, copies):
    files_to_copy = {}
    hf_api = HfApi(endpoint=endpoint, token=token)
    for src_revision, operations in itertools.groupby(copies, key=lambda op: op.src_revision):
        paths = [op.src_path_in_repo for op in operations]
        src_repo_files = hf_api.list_files_info(repo_id=repo_id, paths=paths, revision=src_revision or revision, repo_type=repo_type)
        for src_repo_file in src_repo_files:
            # 非lfs文件'lfs'的取值为None
            # RepoFile:
            #     {
            #         'blob_id': '600c48de5e368ec6822ebddfccab8f3405913ebc',
            #         'lastCommit': {
            #             'date': '2023-09-27T09:49:04.000Z',
            #             'id': 'f9ec1fee0dec37acd83a61a8e975175b41a20149',
            #             'title': 'Upload test/bin_sample.bin with huggingface_hub'
            #         },
            #         'lfs': {
            #             'pointer_size': 127,
            #             'sha256': '4252f8d56b4bb236d0b1bc95a1202e392ca84ce0644bf628398fbb9517287da8',
            #             'size': 12
            #         },
            #         'rfilename': 'test/bin_sample.bin',
            #         'security': {
            #             'avScan': {'virusFound': False, 'virusNames': None},
            #             'blobId': '600c48de5e368ec6822ebddfccab8f3405913ebc',
            #             'name': 'test/bin_sample.bin',
            #             'pickleImportScan': None,
            #             'safe': True},
            #         'size': 12
            #     }
            assert src_repo_file.lfs, "Copying a non-LFS file is not implemented"
            files_to_copy[(src_repo_file.rfilename, src_revision)] = src_repo_file

        for operation in operations:
            if (operation.src_path_in_repo, src_revision) not in files_to_copy:
                raise ValueError(f"Cannot copy {operation.src_path_in_repo} at revision {src_revision or revision}: file is missing on repo.")
    return files_to_copy

# ============== step 3: upload_lfs_files (实际执行 additions 中的lfs文件上传)========
def _upload_single_part(operation: "CommitOperationAdd", upload_url: str) -> None:
    with operation.as_file(with_tqdm=True) as fileobj:
        # 原始实现包含重试机制, 因此在重试时需要使用fileobj.tell与seek函数重置IO的状态
        # response = http_backoff("PUT", upload_url, data=fileobj)
        response = get_session().put(upload_url, data=fileobj)
        # hf_raise_for_status(response)  # 确保请求成功

def _upload_multi_part(operation: "CommitOperationAdd", header: Dict, chunk_size: int, upload_url: str):
    # 备注: 此实现是 HF_HUB_ENABLE_HF_TRANSFER 环境变量为 0 的情况, 将其设置为 1, 则会触发 hf_transfer 的实现
    # hf_transfer: https://github.com/huggingface/hf_transfer
    sorted_parts_urls = [upload_url for part_num, upload_url in header.items()]  # 按 int(part_num) 排序, 此处略去这个逻辑
    
    # step 1: 分块上传
    headers = []
    with operation.as_file(with_tqdm=True) as fileobj:
        for part_idx, part_upload_url in enumerate(sorted_parts_urls):
            with SliceFileObj(fileobj, seek_from=chunk_size * part_idx, read_limit=chunk_size) as fileobj_slice:
                # 原始实现包含重试机制, 因此在重试时需要使用fileobj.tell与seek函数重置IO的状态
                # part_upload_res = http_backoff("PUT", part_upload_url, data=fileobj_slice)
                part_upload_res = get_session().put(part_upload_url, data=fileobj_slice)
                # hf_raise_for_status(part_upload_res)
                headers.append(part_upload_res.headers)
    
    # step 2: 验证上传成功
    parts = []
    for part_number, header in enumerate(headers):
        etag = header.get("etag")
        if etag is None or etag == "":
            raise ValueError(f"Invalid etag (`{etag}`) returned for part {part_number + 1}")
        parts.append(
            {
                "partNumber": part_number + 1,
                "etag": etag,
            }
        )
    completion_json = {"oid": operation.upload_info.sha256.hex(), "parts": parts}
    LFS_HEADERS = {
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    }
    completion_res = get_session().post(
        upload_url,
        json=completion_json,
        headers=LFS_HEADERS,
    )
    # hf_raise_for_status(completion_res)
    

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
        # 上传成功, verify_resp.code == 200; verify_resp.text="OK";
        # 上传失败, verify_resp.code=404, verify_resp.json()={"error": No file uploaded for oid: e375926004d08b0495f07d77826e0642a50850a0eac18ad7e6c6a7f350836669"}
        assert verify_resp.code == 200

def upload_lfs_files(endpoint, repo_id, repo_type, revision, token, create_pr, lfs_additions):
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
        # 入参hex之后的值(与回参相同): [
        # '4252f8d56b4bb236d0b1bc95a1202e392ca84ce0644bf628398fbb9517287da8',
        # 'e375926004d08b0495f07d77826e0642a50850a0eac18ad7e6c6a7f350836669'
        # ]

        objects = response_json['objects']
        # !!!! 注意这个 oid 与 .git/object 目录中的不同 !!!!
        # response_json
        # {
        #     'transfer': 'basic',
        #     'objects': [
        #         {'oid': '4252f8d56b4bb236d0b1bc95a1202e392ca84ce0644bf628398fbb9517287da8', 'size': 12},
        #         {
        #             'oid': 'e375926004d08b0495f07d77826e0642a50850a0eac18ad7e6c6a7f350836669',
        #             'size': 373,
        #             'authenticated': True,
        #             'actions': {
        #                 'upload': {
        #                     'href': 'https://s3.us-east-1.amazonaws.com/lfs.huggingface.co/repos/e2/d8/e2d863e496153629cff5c87e2c54a601f6f2f1a38c316c0639467f57d281643d/e375926004d08b0495f07d77826e0642a50850a0eac18ad7e6c6a7f350836669?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIA4N7VTDGO27GPWFUO%2F20230928%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230928T062445Z&X-Amz-Expires=900&X-Amz-Signature=ba508657e53f75e674904c7752757c3e7e0a41cbe04880497e6e77fe9e8b4cdc&X-Amz-SignedHeaders=host&x-amz-storage-class=INTELLIGENT_TIERING&x-id=PutObject'
        #                 },
        #                 'verify': {
        #                     'href': 'https://huggingface.co/Buxian/test-model.git/info/lfs/objects/verify',
        #                     'header': {'Authorization': 'Basic YWNjZXNzX3Rva2VuOmhmX3VFS3ppWGZWRGhNTVBmdUREekJQek1IYndWTU5GY0N2ZWo='}
        #                 }
        #             }
        #         }
        #     ]
        # }

        # 推测对于大文件(几十个GB), actions.upload 还包含一个 "header" 的键, 值是: {"0": "url_0", "1": "url_1"}
        assert all(["error" not in obj for obj in objects])
        batch_actions += objects

    oid2addop = {add_op.upload_info.sha256.hex(): add_op for add_op in lfs_additions}
    batch_actions = [a for a in batch_actions if a.get("actions") is not None]

    # 此处可以并行实现, 也可以设置 os.environ['HF_HUB_ENABLE_HF_TRANSFER']="1" 更高效传输
    for batch_action in batch_actions:
        operation = oid2addop[batch_action["oid"]]
        # lfs_upload 只负责单个文件上传, operation 必然是 CommitOperationAdd 且必然是大文件
        lfs_upload(operation=operation, lfs_batch_action=batch_action, token=token)

# ============== step 4: prepare_commit_payload (实际执行 additions 中非lfs文件上传, copies 中 lfs 拷贝, delete文件删除) ========
def prepare_commit_payload(
    operations, upload_modes, files_to_copy,
    commit_message, commit_description, parent_commit
    ):
    # 1. Send a header item with the commit metadata
    header_value = {"summary": commit_message, "description": commit_description}
    if parent_commit is not None:
        header_value["parentCommit"] = parent_commit
    yield {"key": "header", "value": header_value}

    # 2. Send operations, one per line
    for operation in operations:
        # 2.a. Case adding a regular file
        if isinstance(operation, CommitOperationAdd) and upload_modes.get(operation.path_in_repo) == "regular":
            yield {
                "key": "file",
                "value": {
                    "content": operation.b64content().decode(),
                    "path": operation.path_in_repo,
                    "encoding": "base64",
                },
            }
        # 2.b. Case adding an LFS file
        elif isinstance(operation, CommitOperationAdd) and upload_modes.get(operation.path_in_repo) == "lfs":
            yield {
                "key": "lfsFile",
                "value": {
                    "path": operation.path_in_repo,
                    "algo": "sha256",
                    "oid": operation.upload_info.sha256.hex(),
                    "size": operation.upload_info.size,
                },
            }
        # 2.c. Case deleting a file or folder
        elif isinstance(operation, CommitOperationDelete):
            yield {
                "key": "deletedFolder" if operation.is_folder else "deletedFile",
                "value": {"path": operation.path_in_repo},
            }
        # 2.d. Case copying a file or folder
        elif isinstance(operation, CommitOperationCopy):
            file_to_copy = files_to_copy[(operation.src_path_in_repo, operation.src_revision)]
            if not file_to_copy.lfs:
                raise NotImplementedError("Copying a non-LFS file is not implemented")
            yield {
                "key": "lfsFile",
                "value": {
                    "path": operation.path_in_repo,
                    "algo": "sha256",
                    "oid": file_to_copy.lfs["sha256"],
                },
            }
        # 2.e. Never expected to happen
        else:
            raise ValueError(
                f"Unknown operation to commit. Operation: {operation}. Upload mode:"
                f" {upload_modes.get(operation.path_in_repo)}"
            )

# ============== step 5: 创建 commit 请求 ===============
def do_commit(endpoint, repo_id, repo_type, revision, token, create_pr, commit_payload):
    # 此方法源码中没有抽象出来
    commit_url = f"{endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"
    headers = {
        "Content-Type": "application/x-ndjson",
        "user-agent": "mylib/v1.0; hf_hub/0.17.2; python/3.9.16; torch/1.12.1+cu113;",
        "authorization": f"Bearer {token}"
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
    # {'success': True, 'commitOid': '003e9ffb13bdb747b8a128abbcb5841964c1a054', 'commitUrl': 'https://huggingface.co/Buxian/test-model/commit/003e9ffb13bdb747b8a128abbcb5841964c1a054', 'hookOutput': ''}

    commit_info = CommitInfo(
        commit_url=commit_data["commitUrl"],
        commit_message=commit_message,
        commit_description=commit_description,
        oid=commit_data["commitOid"],
        pr_url=commit_data["pullRequestUrl"] if create_pr else None,
    )
    return commit_info

if __name__ == "__main__":
    os.environ['HTTP_PROXY'] = "http://172.18.48.1:7890"
    os.environ['HTTPS_PROXY'] = "http://172.18.48.1:7890"

    endpoint = "https://huggingface.co"

    token = "hf_ssddddd"  # modify this!!!
    repo_id = "Buxian/test-model"
    repo_type = "model"
    revision = "dev"
    create_pr = False
    commit_message = "commit title"
    commit_description = "commit details"
    parent_commit = None

    # ======== 各类操作 =========
    copies = [
        # 提交前远程仓库里test/bin_sample.bin是lfs文件(.gitattributes), 而test/bin_sample_main.txt原本不是
        # 提交时会修改.gitattributes文件, 添加test/bin_sample_main.txt, 将其视作lfs文件
        CommitOperationCopy(
            src_path_in_repo="test/bin_sample.bin",
            src_revision="main",
            path_in_repo="test/bin_sample_main.txt",
        ),
        CommitOperationCopy(
            src_path_in_repo="test/bin_sample.bin",
            src_revision="main",
            path_in_repo="test/sample_from_main_copy.bin",
        ),
    ]

    additions = [
        CommitOperationAdd(
            path_or_fileobj = "xyz.txt",
            path_in_repo="bin/xyz.bin",       # 文件后缀是.bin, 上传时会将其视为lfs文件
        ),
        CommitOperationAdd(
            path_or_fileobj = "lru_test.py",
            path_in_repo="bin/lru_test.bin",  # 文件后缀是.bin, 上传时会将其视为lfs文件
        ),
        CommitOperationAdd(
            path_or_fileobj = "xyz.txt",
            path_in_repo="bin/xyz.txt",        # 文件后缀是.txt, 上传时会将其视为普通文件
        ),
    ]

    deletes = [
        CommitOperationDelete(
            path_in_repo="c",
            is_folder=True
        )
    ]
    operations = copies + additions + deletes
    create_commit(
        endpoint=endpoint,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
        create_pr=create_pr,
        operations=operations,
        commit_message=commit_message,
        commit_description=commit_description,
        parent_commit=parent_commit
    )

    
