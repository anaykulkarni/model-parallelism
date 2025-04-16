from mpi4py import MPI

class MPIWrapper:
    """
    Wrapper for MPI communication in distributed MoE
    """

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def get_rank(self):
        """Get the rank of current process"""
        return self.rank

    def get_size(self):
        """Get the total number of processes"""
        return self.size

    def alltoall(self, send_data):
        """All-to-all communication"""
        return self.comm.alltoall(send_data)

    def allgather(self, data):
        """Gather data from all processes to all processes"""
        return self.comm.allgather(data)
    
    def bcast(self, data, root=0):
        """Broadcast data from root to all processes"""
        return self.comm.bcast(data, root=root)
    
    def reduce(self, data, root=0):
        """Reduce data from all processes to root process"""
        return self.comm.reduce(data, root=root)
    
    def gather(self, data, root=0):
        """Gather data from all processes to root process"""
        return self.comm.gather(data, root=root)
    
    def scatter(self, data, root=0):
        """Scatter data from root process to all processes"""
        return self.comm.scatter(data, root=root)
    
    def allreduce(self, data):
        """All-reduce data from all processes to all processes"""
        return self.comm.allreduce(data)
    
    def barrier(self):
        """Synchronization barrier"""
        self.comm.barrier()


# Global MPI wrapper
mpi = MPIWrapper()
