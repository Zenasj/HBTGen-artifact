python
import os
import urllib.parse
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
    NamedTuple
)
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import datetime
import time

def _fetch_url(url: str, *,
               headers: Optional[Dict[str, str]] = None,
               data: Optional[Dict[str, Any]] = None,
               method: Optional[str] = None,
               reader: Callable[[Any], Any] = lambda x: x.read()) -> Any:
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith('https://api.github.com/'):
        headers['Authorization'] = f'token {token}'
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return reader(conn)
    except HTTPError as err:
        print(err.reason)
        print(err.headers)
        if err.code == 403 and all(key in err.headers for key in ['X-RateLimit-Limit', 'X-RateLimit-Used']):
            print(f"Rate limit exceeded: {err.headers['X-RateLimit-Used']}/{err.headers['X-RateLimit-Limit']}")
        raise

def fetch_json(url: str,
               params: Optional[Dict[str, Any]] = None,
               data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if params is not None and len(params) > 0:
        url += '?' + '&'.join(f"{name}={urllib.parse.quote(str(val))}" for name, val in params.items())
    return cast(List[Dict[str, Any]], _fetch_url(url, headers=headers, data=data, reader=json.load))

def main():

    FILTER_OUT_USERS = set(["pytorchmergebot", "facebook-github-bot"])

    period_begin_date = datetime.date(2022, 6, 5)
    #period_begin_date = datetime.date(2022, 11, 20)

    while period_begin_date <= datetime.date(2022, 12, 11):
        period_end_date = period_begin_date + datetime.timedelta(days=6)

        response = cast(
            Dict[str, Any],
            fetch_json(
                "https://api.github.com/search/issues",
                params={"q": f'repo:pytorch/pytorch is:pr is:closed label:"open source" label:Merged -label:Reverted closed:{period_begin_date}..{period_end_date}', "per_page": '100'},
            ),
        )

        pr_count = 0
        users = set([])
        for item in response["items"]:
            u = item["user"]["login"]
            if u not in FILTER_OUT_USERS:
                pr_count += 1
                users.add(u)

        user_count = len(users)
        print(f'{period_begin_date},{period_end_date},{pr_count},{user_count}')
        #print(users)
        period_begin_date = period_end_date + datetime.timedelta(days=1)
        time.sleep(10)

if __name__ == "__main__":
    main()