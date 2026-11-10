import os

from tabpfn_client import set_access_token

if "TABPFN_TOKEN" not in os.environ:
    os.environ["TABPFN_TOKEN"] = "tabpfn_sk_xzjB48FcS4o6DXzkU2oVkbDr66i9nDRq2Y4rVs6XP8A"
    # raise ValueError("TABPFN_TOKEN not set")
    set_access_token(os.environ["TABPFN_TOKEN"])
else:
    set_access_token(os.environ["TABPFN_TOKEN"])
