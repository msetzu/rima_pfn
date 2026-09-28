import os

from tabpfn_client import set_access_token

if "TABPFN_TOKEN" not in os.environ:
    raise ValueError("TABPFN_TOKEN not set")
else:
    set_access_token(os.environ["TABPFN_TOKEN"])
