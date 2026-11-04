"""Which device each rank's TensorFlow uses (docs/members.md, "On GPUs").

Only rank 0 and the ranks that train members (training.members) run TensorFlow. Those take one GPU
each, in turn over the GPUs of their node by their place among the TensorFlow processes there (not
among all its ranks: the member ranks are spread at an even stride, 8 on 96 ranks, so every one of
them would take GPU 0 of two), with memory growth on: TensorFlow
otherwise reserves nearly all of a GPU's memory in the first process that uses it, and the members
started after it on the same GPU fail. Every other rank hides the GPUs from itself before the
likelihood is built, so that a likelihood code importing TensorFlow or JAX takes none.
training.member_device decides for the members: auto (a GPU when the member's process sees one,
else the CPU), cpu or gpu (stop when there is none). Rank 0 uses a GPU whenever it sees one.

set_up_devices is collective (every rank calls it once, before the likelihood); set_up_tensorflow is
called by a process before its first use of TensorFlow and is a no-op afterwards.
"""
import os

DEVICES = ("auto", "cpu", "gpu")
DEVICE = None       # what this process's TensorFlow computes on, once set_up_tensorflow has run
GPU_SLOT = 0        # this process's place among the TensorFlow processes of its node (set_up_devices)


def keeps_gpus(rank, size, members, member_device, boosts):
    """Whether rank `rank` of `size` keeps the node's GPUs visible: rank 0 always; a rank that
    trains members (training.members > 1 with boosts, member_device not cpu); no other."""
    if rank == 0:
        return True
    if members <= 1 or not boosts or member_device == "cpu":
        return False
    from training.members import member_ranks

    return rank in member_ranks(size, members)


def gpu_index(slot, n_gpus):
    """The GPU a TensorFlow process takes: the TensorFlow processes of a node in turn over its GPUs."""
    return slot % n_gpus


def gpu_slot(rank, ranks_on_node, keeps):
    """Rank `rank`'s place, in rank order, among the ranks of its node that run TensorFlow (`keeps`
    true); 0 for a rank that runs none."""
    tensorflow = sorted(r for r in ranks_on_node if keeps(r))
    return tensorflow.index(rank) if rank in tensorflow else 0


def node_ranks(comm):
    """The ranks of this rank's node (collective)."""
    if comm is None or comm.Get_size() == 1:
        return [0 if comm is None else comm.Get_rank()]
    from mpi4py import MPI

    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        return node.allgather(comm.Get_rank())
    finally:
        node.Free()


def set_up_devices(cfg, comm, rank):
    """Every rank, before the likelihood is built: its place among the TensorFlow processes of its
    node, and the GPUs hidden from a rank that does not run TensorFlow."""
    global GPU_SLOT
    size = 1 if comm is None else comm.Get_size()
    boosts = cfg.acquisition.boost is not None

    def keeps(r):
        return keeps_gpus(r, size, cfg.training.members, cfg.training.member_device, boosts)

    GPU_SLOT = gpu_slot(rank, node_ranks(comm), keeps)
    if not keeps(rank):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""


def _current():
    """What an already initialised TensorFlow computes on in this process."""
    import tensorflow as tf

    gpus = tf.config.list_logical_devices("GPU")
    return describe_gpu(gpus[0].name) if gpus else _describe_cpu(tf)


def _describe_cpu(tf):
    threads = tf.config.threading.get_intra_op_parallelism_threads()
    return f"CPU ({threads} threads)" if threads else "CPU"


def describe_gpu(name, details=None):
    model = (details or {}).get("device_name")
    return f"GPU {name.rsplit(':', 1)[-1]}" + (f" ({model})" if model else "")


def set_up_tensorflow(device="auto", threads=None):
    """This process's TensorFlow, before its first use: the threads, and the device (one GPU with
    memory growth, or the CPU). Returns what it computes on; later calls return the same."""
    global DEVICE
    if device not in DEVICES:
        raise ValueError(f"device {device!r}: one of {', '.join(DEVICES)}")
    if DEVICE is not None:
        return DEVICE
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("training.member_device: gpu, and this process sees no GPU")
    try:
        if threads:
            tf.config.threading.set_intra_op_parallelism_threads(int(threads))
            tf.config.threading.set_inter_op_parallelism_threads(1)
        if gpus and device != "cpu":
            gpu = gpus[gpu_index(GPU_SLOT, len(gpus))]
            tf.config.set_visible_devices([gpu], "GPU")
            tf.config.experimental.set_memory_growth(gpu, True)
            DEVICE = describe_gpu(gpu.name, tf.config.experimental.get_device_details(gpu))
        else:
            if gpus:
                tf.config.set_visible_devices([], "GPU")
            DEVICE = _describe_cpu(tf)
    except RuntimeError:                       # TensorFlow already initialised in this process
        DEVICE = _current()
    return DEVICE
