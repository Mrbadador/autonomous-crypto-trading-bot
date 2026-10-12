import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
import logging

log = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False
    log.warning(
        "[DARK-FOREST] torch_geometric not installed. "
        "DarkForestGNN whale-flow module will run in stub mode. "
        "Install with: pip install torch_geometric"
    )

class MempoolGraphBuilder:
    """Builds transaction graph from mempool data."""
    def __init__(self, binance_hot_wallets, max_nodes=5000):
        self.hot_wallets = set(binance_hot_wallets)
        self.max_nodes = max_nodes
        self.addr_to_idx = {}
        self.idx_to_addr = {}
        
    def build_graph(self, transactions, current_time):
        if not _TORCH_GEOMETRIC_AVAILABLE:
            return None
        self.addr_to_idx.clear()
        self.idx_to_addr.clear()
        active_addrs = set()
        
        for tx in transactions:
            active_addrs.add(tx['from'])
            active_addrs.add(tx['to'])
            
        for i, addr in enumerate(list(active_addrs)[:self.max_nodes]):
            self.addr_to_idx[addr] = i
            self.idx_to_addr[i] = addr
            
        n_nodes = len(self.addr_to_idx)
        if n_nodes == 0:
            return None

        x = np.zeros((n_nodes, 6))
        addr_balances = defaultdict(lambda: {'USDT': 0, 'USDC': 0, 'ETH': 0})
        
        for tx in transactions:
            if tx['to'] in self.addr_to_idx:
                addr_balances[tx['to']][tx['token']] += tx['value']
                
        for addr, idx in self.addr_to_idx.items():
            x[idx, 0] = addr_balances[addr]['USDT']
            x[idx, 1] = addr_balances[addr]['USDC']
            x[idx, 2] = addr_balances[addr]['ETH']
            x[idx, 5] = 1.0 if addr in self.hot_wallets else 0.0
            
        edge_list, edge_attrs = [], []
        
        for tx in transactions:
            if tx['from'] in self.addr_to_idx and tx['to'] in self.addr_to_idx:
                src = self.addr_to_idx[tx['from']]
                dst = self.addr_to_idx[tx['to']]
                
                age = current_time - tx['timestamp']
                time_decay = 0.99 ** (age / 60)
                weight = np.log(tx['value'] + 1) * time_decay
                token_type = {'USDT': 0, 'USDC': 1, 'ETH': 2}.get(tx['token'], 3)
                
                edge_list.append([src, dst])
                edge_attrs.append([weight, token_type, tx['value']])
                
        if not edge_list:
            return None

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        x_tensor = torch.tensor(x, dtype=torch.float)
        
        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

if _TORCH_GEOMETRIC_AVAILABLE:
    class DarkForestGNN(nn.Module):
        """Graph Attention Network for Whale Flow Prediction."""
        def __init__(self, in_channels=6, edge_channels=3, hidden_channels=64, heads=4):
            super().__init__()
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=False, edge_dim=edge_channels)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False, edge_dim=edge_channels)
            
            self.centrality_mlp = nn.Sequential(nn.Linear(hidden_channels, 32), nn.ReLU(), nn.Linear(32, 1))
            self.flow_mlp = nn.Sequential(nn.Linear(hidden_channels * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            
        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            
            x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
            node_emb = self.conv2(x, edge_index, edge_attr)
            
            centrality = self.centrality_mlp(node_emb).squeeze(-1)
            
            src_emb = node_emb[edge_index[0]]
            dst_emb = node_emb[edge_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
            flow_prob = self.flow_mlp(edge_emb).squeeze(-1)
            
            return {'node_embedding': node_emb, 'centrality': centrality, 'flow_probability': flow_prob}
else:
    class DarkForestGNN:
        """Stub class when torch_geometric is not available."""
        def __init__(self, *args, **kwargs):
            log.warning("[DARK-FOREST] DarkForestGNN running as stub (torch_geometric unavailable).")

        def forward(self, data):
            return {'node_embedding': None, 'centrality': None, 'flow_probability': None}

        def __call__(self, data):
            return self.forward(data)

class SyntheticMempool:
    """Generates fake on-chain data for testing until an RPC node is integrated."""
    @staticmethod
    def get_recent_transactions():
        current_time = time.time()
        txs = []
        # Simulate normal noise
        for _ in range(50):
            txs.append({
                'from': f"0x{np.random.randint(1000, 9999)}",
                'to': f"0x{np.random.randint(1000, 9999)}",
                'value': np.random.uniform(100, 10000),
                'token': 'USDT', 'timestamp': current_time - np.random.uniform(0, 60)
            })
        # Simulate a whale moving 5M USDT to a Binance Hot Wallet
        txs.append({
            'from': "0xWHALE_WALLET_99", 'to': "0xBINANCE_HOT_1",
            'value': 5000000.0, 'token': 'USDT', 'timestamp': current_time - 2
        })
        return txs
