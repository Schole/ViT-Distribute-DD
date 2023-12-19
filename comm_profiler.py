import torch
import torch.distributed as dist
import time

class CommunicationProfiler:
    def __init__(self, world_size, gpus_per_node):
        self.world_size = world_size
        self.gpus_per_node = gpus_per_node
        self.intra_node_data = 0
        self.inter_node_data = 0
        self.intra_node_comm_time = 0
        self.inter_node_comm_time = 0

    def communication_start(self, is_intra_node):
        self.start_time = time.time()
        self.is_intra_node = is_intra_node

    def communication_end(self, size):
        end_time = time.time()
        comm_time = end_time - self.start_time

        if self.is_intra_node:
            self.intra_node_data += size
            self.intra_node_comm_time += comm_time
        else:
            self.inter_node_data += size
            self.inter_node_comm_time += comm_time

    def is_intra_node_comm(self, target_rank):
        node_rank = dist.get_rank() // self.gpus_per_node
        target_node_rank = target_rank // self.gpus_per_node
        return node_rank == target_node_rank

    def get_stats(self):
        return self.intra_node_data, self.inter_node_data, self.intra_node_comm_time, self.inter_node_comm_time

