{
  "target_repo": "pytorch",
  "result": [
    {
      "commit1": "91ead3eae4c",
      "commit1_time": "2024-03-21 01:56:42 +0000",
      "commit1_digest": {
        "name": "test_bench",
        "environ": {
          "pytorch_git_version": "91ead3eae4cd6cbf50fe7a7b4a2f9f35302bc9b2",
          "pytorch_version": "2.4.0a0+git91ead3e",
          "device": "NVIDIA A100-SXM4-40GB",
          "git_commit_hash": "91ead3eae4cd6cbf50fe7a7b4a2f9f35302bc9b2"
        },
        "metrics": {
          "model=sam_fast, test=eval, device=cuda, bs=None, extra_args=['--memleak'], metric=memleak": "False"
        }
      },
      "commit2": "e9dcda5cba9",
      "commit2_time": "2024-03-21 01:57:08 +0000",
      "commit2_digest": {
        "name": "test_bench",
        "environ": {
          "pytorch_git_version": "e9dcda5cba92884be6432cf65a777b8ed708e3d6",
          "pytorch_version": "2.4.0a0+gite9dcda5",
          "device": "NVIDIA A100-SXM4-40GB",
          "git_commit_hash": "e9dcda5cba92884be6432cf65a777b8ed708e3d6"
        },
        "metrics": {
          "model=sam_fast, test=eval, device=cuda, bs=None, extra_args=['--memleak'], metric=memleak": "True"
        }
      }
    }
  ]
}