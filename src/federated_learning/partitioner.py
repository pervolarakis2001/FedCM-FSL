from dataclasses import dataclass, field
import copy
import numpy as np
import pandas as pd


from dataclasses import dataclass, field
import copy


@dataclass
class ClientPartition:
    client_id: int
    df: pd.DataFrame
    has_s2: bool = True
    has_s1: bool = True


# ── Modality assignment


def _assign_unimodal(partitions: list[ClientPartition]) -> list[ClientPartition]:
    """
    Splits clients into two equal unimodal halves:
      - First  half → SAR only        (has_s1=True,  has_s2=False)
      - Second half → Sentinel-2 only (has_s1=False, has_s2=True)
    Works for any even or odd number of clients (odd → S1 gets the extra one).
    """
    n_half = len(partitions) // 2
    for p in partitions[:n_half]:  # first half  → S1 / SAR only
        p.has_s1 = True
        p.has_s2 = False
    for p in partitions[n_half:]:  # second half → S2 only
        p.has_s1 = False
        p.has_s2 = True
    return partitions


# ── IID split


def _iid_split(
    df: pd.DataFrame,
    n_clients: int,
    seed: int = 42,
    unimodal: bool = False,  # ← new
) -> list[ClientPartition]:
    working = df[df["season"] == "Summer"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(working))
    rng.shuffle(idx)

    partitions = [
        ClientPartition(
            client_id=i,
            df=working.iloc[split].reset_index(drop=True),
        )
        for i, split in enumerate(np.array_split(idx, n_clients))
    ]
    return _assign_unimodal(partitions) if unimodal else partitions


# ── Non-IID split
def _non_iid_split(
    df: pd.DataFrame,
    n_clients: int,
    unimodal: bool = False,  # ← new
) -> list[ClientPartition]:
    countries = sorted(df["country"].unique())
    n_countries = len(countries)

    if n_clients >= n_countries:
        partitions = [
            ClientPartition(
                client_id=i,
                df=df[df["country"] == c].reset_index(drop=True),
            )
            for i, c in enumerate(countries)
        ]
    else:
        sizes = {c: int((df["country"] == c).sum()) for c in countries}
        buckets: list[list[str]] = [[] for _ in range(n_clients)]
        loads = [0] * n_clients
        for country in sorted(countries, key=lambda c: -sizes[c]):
            lightest = int(np.argmin(loads))
            buckets[lightest].append(country)
            loads[lightest] += sizes[country]
        partitions = [
            ClientPartition(
                client_id=i,
                df=df[df["country"].isin(bucket)].reset_index(drop=True),
            )
            for i, bucket in enumerate(buckets)
        ]

    return _assign_unimodal(partitions) if unimodal else partitions
