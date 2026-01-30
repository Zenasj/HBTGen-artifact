{
    "id": comment_id,
    "pr_num": pr_num,
    "owner": owner,
    "project": project,
    "pending_checks": pending_checks,  # At the time of the merge
    "failed_checks": failed_checks,  # At the time of the merge
    "is_failed": is_failed,  # This is set to True if the merge fails to get through for whatever reason
    "dry_run": dry_run,
    "skip_mandatory_checks": skip_mandatory_checks,
    "ignore_current": ignore_current,
    "error": error,  # The same Exception message that will be shown on PR
}

{
  "_id": "52d3152b-ec35-4b5a-91fc-0e7298fc54b5-1",
  "_event_time": "2023-03-23T21:10:32.754368Z",
  "_meta": null,
  "owner": "pytorch",
  "is_failed": true,
  "id": 1478678477,
  "failed_checks": [],
  "dry_run": true,
  "error": "Command `git -C pytorch cherry-pick -x cc0d2e0fba648bb5deda34a9056f2c4192b22314` returned non-zero exit code 1...",
  "ignore_current": false,
  "project": "pytorch",
  "pr_num": 97293,
  "skip_mandatory_checks": false,
  "pending_checks": []
}

{
  "_id": "dd7d2580-f6e5-47e7-9441-17df86056c14-1",
  "_event_time": "2023-03-23T21:43:53.915911Z",
  "_meta": null,
  "owner": "pytorch",
  "is_failed": true,
  "id": 1481949104,
  "failed_checks": [],
  "dry_run": true,
  "error": "PR #97471 has not been reviewed yet",
  "ignore_current": false,
  "project": "pytorch",
  "pr_num": 97471,
  "skip_mandatory_checks": true,
  "pending_checks": []
}

{
  "_id": "5d7de4e3-1af1-4869-a3b7-d1a9dbced6ce-1",
  "_event_time": "2023-03-24T00:10:41.914111Z",
  "_meta": null,
  "is_failed": false,
  "id": 1481949104,
  "failed_checks": [],
  "error": "",
  "last_commit_sha": "4657400513f0360a0a4f73d46e1aff0882221687",
  "merge_commit_sha": "416bac5b813a181753afade781ae30f4f0843586",
  "ignore_current": false,
  "pending_checks": [
    [
      "pull / linux-focal-py3.8-gcc7 / test (default, 1, 3, linux.2xlarge)",
      "https://github.com/pytorch/pytorch/actions/runs/4506464828/jobs/7933518379",
      12239935788
    ],
    ...
    [
      "trunk / linux-bionic-cuda11.8-py3.10-gcc7 / test (default, 5, 5, linux.4xlarge.nvidia.gpu)",
      "https://github.com/pytorch/pytorch/actions/runs/4506465633/jobs/7933621958",
      12240067113
    ],
    ...
  ],
  "owner": "pytorch",
  "skip_mandatory_checks": true,
  "author": "Huy Do <huydhn@gmail.com>",
  "project": "pytorch",
  "merge_base_sha": "a3b30c5025e3381022fa00b127b0d881f4ef66d4",
  "pr_num": 97471
}