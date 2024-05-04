import gzip
import os
from typing import Dict

import numpy as np
import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.data.data import BaseData, Data
from tqdm import tqdm


class SteamDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.data_len = 88310
        self.user_ids = set()
        self.game_ids = set()
        self.edges = list()
        super(SteamDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["australian_users_items.json.gz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self) -> None:
        url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/steam/australian_users_items.json.gz"
        path = download_url(url, self.raw_dir)
        return

    def process(self) -> None:
        g = gzip.open(self.raw_paths[0], "r")

        for item in tqdm(g, total=self.data_len):
            parsed_item = eval(item)
            self._process_data(parsed_item)

        combined_nodes = list(self.user_ids) + list(self.game_ids)

        edge_index = torch.empty((2, len(self.edges)), dtype=torch.long)
        x = torch.empty((len(combined_nodes), 1))
        self.mapping = dict(zip(combined_nodes, range(len(combined_nodes))))

        for i, (src, dst) in enumerate(self.edges):
            edge_index[0, i] = self.mapping[src]
            edge_index[1, i] = self.mapping[dst]

        for i, node_id in enumerate(combined_nodes):
            x[self.mapping[node_id], 0] = 0 if node_id.startswith("u") else 1

        data = Data(x=x, edge_index=edge_index, y=x)

        torch.save(data, self.processed_paths[0])

    def _process_data(self, data: Dict):
        user_id = data["steam_id"]
        self.user_ids.add("u" + str(user_id))
        for game in data["items"]:
            game_id = game["item_id"]
            playtime = game["playtime_forever"]
            # Introduce threshold playtime
            if int(playtime) > 5000:
                self.game_ids.add("g" + game_id)
                self.edges.append(("u" + str(user_id), "g" + str(game_id)))

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        data = torch.load(os.path.join(self.processed_dir, f"data.pt"))
        return data
