
from dataclasses import dataclass, field
import copy
import numpy as np
import pandas as pd
from config import PAPER_COUNTRIES, DS2_COUNTRIES


@dataclass
class ClientPartition:
    client_id: int
    df: pd.DataFrame
    has_s2: bool = True
    has_s1: bool = True


def partition_by_scenario(df, scenario, n_clients, seed=42):
    rng = np.random.default_rng(seed)

    if scenario == "DS2":
        working = df[
            (df["season"] == "Summer") &
            (df["country"].isin(DS2_COUNTRIES))
        ].reset_index(drop=True)
        partitions = _country_split(working, len(DS2_COUNTRIES))

    elif scenario == "DS3":
        working = df[
            df["country"].isin(PAPER_COUNTRIES)
        ].reset_index(drop=True)
        partitions = _country_split(working, len(PAPER_COUNTRIES))

    else:  # DS1, DS4, DS5
        working = df[df["season"] == "Summer"].reset_index(drop=True)  # ← key line
        idx = np.arange(len(working))
        rng.shuffle(idx)
        partitions = [
            ClientPartition(client_id=i, df=working.iloc[s].reset_index(drop=True))
            for i, s in enumerate(np.array_split(idx, n_clients))
        ]

    n_half = len(partitions) // 2
    if scenario == "DS4":
        for p in partitions[:n_half]: p.has_s2 = False
    elif scenario == "DS5":
        for p in partitions[:n_half]: p.has_s1 = False

    for p in partitions:
        if len(p.df) == 0:
            raise ValueError(f"Client {p.client_id} got zero samples.")
    return partitions


def _country_split(df: pd.DataFrame, n_clients: int) -> list[ClientPartition]:
    """
    One client per country. If n_clients < n_countries, countries are
    greedily bucketed by size (largest-first) to balance client loads.
    """
    countries = sorted(df["country"].unique())

    if n_clients >= len(countries):
        return [
            ClientPartition(
                client_id=i, df=df[df["country"] == c].reset_index(drop=True)
            )
            for i, c in enumerate(countries)
        ]

    # Greedy bin-packing: assign countries to the lightest bucket
    sizes = {c: (df["country"] == c).sum() for c in countries}
    buckets = [[] for _ in range(n_clients)]
    loads = [0] * n_clients
    for c in sorted(countries, key=lambda c: -sizes[c]):
        b = int(np.argmin(loads))
        buckets[b].append(c)
        loads[b] += sizes[c]

    return [
        ClientPartition(
            client_id=i, df=df[df["country"].isin(b)].reset_index(drop=True)
        )
        for i, b in enumerate(buckets)
    ]
