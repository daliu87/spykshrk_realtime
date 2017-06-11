# encoding: utf-8
# module mpi4py.MPI
# from /home/daliu/anaconda3/lib/python3.5/site-packages/mpi4py/MPI.cpython-35m-x86_64-linux-gnu.so
# by generator 1.145
""" Message Passing Interface """

# imports
import builtins as __builtins__ # <module 'builtins' (built-in)>

# Variables with simple values

ANY_SOURCE = -1
ANY_TAG = -1

APPNUM = 4

BOTTOM = 0

BSEND_OVERHEAD = 128

CART = 1

COMBINER_CONTIGUOUS = 2
COMBINER_DARRAY = 13
COMBINER_DUP = 1

COMBINER_F90_COMPLEX = 15
COMBINER_F90_INTEGER = 16
COMBINER_F90_REAL = 14

COMBINER_HINDEXED = 8

COMBINER_HINDEXED_BLOCK = 18

COMBINER_HVECTOR = 5
COMBINER_INDEXED = 6

COMBINER_INDEXED_BLOCK = 9

COMBINER_NAMED = 0
COMBINER_RESIZED = 17
COMBINER_STRUCT = 11
COMBINER_SUBARRAY = 12
COMBINER_VECTOR = 3

COMM_TYPE_SHARED = 0

CONGRUENT = 1

DISPLACEMENT_CURRENT = -54278278

DISP_CUR = -54278278

DISTRIBUTE_BLOCK = 0
DISTRIBUTE_CYCLIC = 1

DISTRIBUTE_DFLT_DARG = -1

DISTRIBUTE_NONE = 2

DIST_GRAPH = 3

ERR_ACCESS = 20
ERR_AMODE = 21
ERR_ARG = 13
ERR_ASSERT = 22

ERR_BAD_FILE = 23

ERR_BASE = 24
ERR_BUFFER = 1
ERR_COMM = 5
ERR_CONVERSION = 25
ERR_COUNT = 2
ERR_DIMS = 12
ERR_DISP = 26

ERR_DUP_DATAREP = 27

ERR_FILE = 30

ERR_FILE_EXISTS = 28

ERR_FILE_IN_USE = 29

ERR_GROUP = 9
ERR_INFO = 34

ERR_INFO_KEY = 31
ERR_INFO_NOKEY = 32
ERR_INFO_VALUE = 33

ERR_INTERN = 17

ERR_IN_STATUS = 18

ERR_IO = 35
ERR_KEYVAL = 36
ERR_LASTCODE = 92
ERR_LOCKTYPE = 37
ERR_NAME = 38

ERR_NOT_SAME = 40

ERR_NO_MEM = 39
ERR_NO_SPACE = 41

ERR_NO_SUCH_FILE = 42

ERR_OP = 10
ERR_OTHER = 16
ERR_PENDING = 19
ERR_PORT = 43
ERR_QUOTA = 44
ERR_RANK = 6

ERR_READ_ONLY = 45

ERR_REQUEST = 7

ERR_RMA_ATTACH = 69
ERR_RMA_CONFLICT = 46
ERR_RMA_FLAVOR = 70
ERR_RMA_RANGE = 68
ERR_RMA_SHARED = 71
ERR_RMA_SYNC = 47

ERR_ROOT = 8
ERR_SERVICE = 48
ERR_SIZE = 49
ERR_SPAWN = 50
ERR_TAG = 4
ERR_TOPOLOGY = 11
ERR_TRUNCATE = 15
ERR_TYPE = 3
ERR_UNKNOWN = 14

ERR_UNSUPPORTED_DATAREP = 51
ERR_UNSUPPORTED_OPERATION = 52

ERR_WIN = 53

GRAPH = 2

HOST = 1

IDENT = 0

IN_PLACE = 1

IO = 2

KEYVAL_INVALID = -1

LASTUSEDCODE = 5

LOCK_EXCLUSIVE = 1
LOCK_SHARED = 2

MAX_DATAREP_STRING = 128

MAX_ERROR_STRING = 256

MAX_INFO_KEY = 36
MAX_INFO_VAL = 256

MAX_LIBRARY_VERSION_STRING = 256

MAX_OBJECT_NAME = 64

MAX_PORT_NAME = 1024

MAX_PROCESSOR_NAME = 256

MODE_APPEND = 128
MODE_CREATE = 1

MODE_DELETE_ON_CLOSE = 16

MODE_EXCL = 64
MODE_NOCHECK = 1
MODE_NOPRECEDE = 2
MODE_NOPUT = 4
MODE_NOSTORE = 8
MODE_NOSUCCEED = 16
MODE_RDONLY = 2
MODE_RDWR = 8
MODE_SEQUENTIAL = 256

MODE_UNIQUE_OPEN = 32

MODE_WRONLY = 4

ORDER_C = 0
ORDER_F = 1
ORDER_FORTRAN = 1

PROC_NULL = -2

ROOT = -4

SEEK_CUR = 602
SEEK_END = 604
SEEK_SET = 600

SIMILAR = 2

SUBVERSION = 0
SUCCESS = 0

TAG_UB = 0

THREAD_FUNNELED = 1
THREAD_MULTIPLE = 3
THREAD_SERIALIZED = 2
THREAD_SINGLE = 0

TYPECLASS_COMPLEX = 3
TYPECLASS_INTEGER = 1
TYPECLASS_REAL = 2

UNDEFINED = -32766
UNEQUAL = 3

UNIVERSE_SIZE = 6

UNWEIGHTED = 2

VERSION = 3

WEIGHTS_EMPTY = 3

WIN_BASE = 7

WIN_CREATE_FLAVOR = 10

WIN_DISP_UNIT = 9

WIN_FLAVOR = 10

WIN_FLAVOR_ALLOCATE = 2
WIN_FLAVOR_CREATE = 1
WIN_FLAVOR_DYNAMIC = 3
WIN_FLAVOR_SHARED = 4

WIN_MODEL = 11
WIN_SEPARATE = 1
WIN_SIZE = 8
WIN_UNIFIED = 0

WTIME_IS_GLOBAL = 3

# functions

def Add_error_class(): # real signature unknown; restored from __doc__
    """
    Add_error_class()
    
        Add an *error class* to the known error classes
    """
    pass

def Add_error_code(errorclass): # real signature unknown; restored from __doc__
    """
    Add_error_code(int errorclass)
    
        Add an *error code* to an *error class*
    """
    pass

def Add_error_string(errorcode, string): # real signature unknown; restored from __doc__
    """
    Add_error_string(int errorcode, string)
    
        Associate an *error string* with an
        *error class* or *errorcode*
    """
    pass

def Alloc_mem(Asize, Info_info=None): # real signature unknown; restored from __doc__
    """
    Alloc_mem(Aint size, Info info=INFO_NULL)
    
        Allocate memory for message passing and RMA
    """
    pass

def Attach_buffer(memory): # real signature unknown; restored from __doc__
    """
    Attach_buffer(memory)
    
        Attach a user-provided buffer for
        sending in buffered mode
    """
    pass

def BAND(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def BOR(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def BXOR(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Close_port(port_name): # real signature unknown; restored from __doc__
    """
    Close_port(port_name)
    
        Close a port
    """
    pass

def Compute_dims(nnodes, dims): # real signature unknown; restored from __doc__
    """
    Compute_dims(int nnodes, dims)
    
        Return a balanced distribution of
        processes per coordinate direction
    """
    pass

def Detach_buffer(): # real signature unknown; restored from __doc__
    """
    Detach_buffer()
    
        Remove an existing attached buffer
    """
    pass

def Finalize(): # real signature unknown; restored from __doc__
    """
    Finalize()
    
        Terminate the MPI execution environment
    """
    pass

def Free_mem(memory): # real signature unknown; restored from __doc__
    """
    Free_mem(memory)
    
        Free memory allocated with `Alloc_mem()`
    """
    pass

def Get_address(location): # real signature unknown; restored from __doc__
    """
    Get_address(location)
    
        Get the address of a location in memory
    """
    pass

def Get_error_class(errorcode): # real signature unknown; restored from __doc__
    """
    Get_error_class(int errorcode)
    
        Convert an *error code* into an *error class*
    """
    pass

def Get_error_string(errorcode): # real signature unknown; restored from __doc__
    """
    Get_error_string(int errorcode)
    
        Return the *error string* for a given
        *error class* or *error code*
    """
    pass

def Get_library_version(): # real signature unknown; restored from __doc__
    """
    Get_library_version()
    
        Obtain the version string of the MPI library
    """
    pass

def Get_processor_name(): # real signature unknown; restored from __doc__
    """
    Get_processor_name()
    
        Obtain the name of the calling processor
    """
    pass

def get_vendor(): # real signature unknown; restored from __doc__
    """
    get_vendor()
    
        Infomation about the underlying MPI implementation
    
        :Returns:
          - a string with the name of the MPI implementation
          - an integer 3-tuple version ``(major, minor, micro)``
    """
    pass

def Get_version(): # real signature unknown; restored from __doc__
    """
    Get_version()
    
        Obtain the version number of the MPI standard supported
        by the implementation as a tuple ``(version, subversion)``
    """
    pass

def Init(): # real signature unknown; restored from __doc__
    """
    Init()
    
        Initialize the MPI execution environment
    """
    pass

def Init_thread(required=None): # real signature unknown; restored from __doc__
    """
    Init_thread(int required=THREAD_MULTIPLE)
    
        Initialize the MPI execution environment
    """
    pass

def Is_finalized(): # real signature unknown; restored from __doc__
    """
    Is_finalized()
    
        Indicates whether ``Finalize`` has completed
    """
    pass

def Is_initialized(): # real signature unknown; restored from __doc__
    """
    Is_initialized()
    
        Indicates whether ``Init`` has been called
    """
    pass

def Is_thread_main(): # real signature unknown; restored from __doc__
    """
    Is_thread_main()
    
        Indicate whether this thread called
        ``Init`` or ``Init_thread``
    """
    pass

def LAND(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Lookup_name(service_name, info=None): # real signature unknown; restored from __doc__
    """
    Lookup_name(service_name, info=INFO_NULL)
    
        Lookup a port name given a service name
    """
    pass

def LOR(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def LXOR(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def MAX(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def MAXLOC(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def MIN(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def MINLOC(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def NO_OP(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Open_port(Info_info=None): # real signature unknown; restored from __doc__
    """
    Open_port(Info info=INFO_NULL)
    
        Return an address that can be used to establish
        connections between groups of MPI processes
    """
    pass

def OP_NULL(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Pcontrol(level): # real signature unknown; restored from __doc__
    """
    Pcontrol(int level)
    
        Control profiling
    """
    pass

def PROD(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Publish_name(service_name, port_name, info=None): # real signature unknown; restored from __doc__
    """
    Publish_name(service_name, port_name, info=INFO_NULL)
    
        Publish a service name
    """
    pass

def Query_thread(): # real signature unknown; restored from __doc__
    """
    Query_thread()
    
        Return the level of thread support
        provided by the MPI library
    """
    pass

def Register_datarep(datarep, read_fn, write_fn, extent_fn): # real signature unknown; restored from __doc__
    """
    Register_datarep(datarep, read_fn, write_fn, extent_fn)
    
        Register user-defined data representations
    """
    pass

def REPLACE(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def SUM(*args, **kwargs): # real signature unknown
    """ Op """
    pass

def Unpublish_name(service_name, port_name, info=None): # real signature unknown; restored from __doc__
    """
    Unpublish_name(service_name, port_name, info=INFO_NULL)
    
        Unpublish a service name
    """
    pass

def Wtick(): # real signature unknown; restored from __doc__
    """
    Wtick()
    
        Return the resolution of ``Wtime``
    """
    pass

def Wtime(): # real signature unknown; restored from __doc__
    """
    Wtime()
    
        Return an elapsed time on the calling processor
    """
    pass

def _addressof(arg): # real signature unknown; restored from __doc__
    """
    _addressof(arg)
    
        Memory address of the underlying MPI handle
    """
    pass

def _handleof(arg): # real signature unknown; restored from __doc__
    """
    _handleof(arg)
    
        Unsigned integer value with the underlying MPI handle
    """
    pass

def _sizeof(arg): # real signature unknown; restored from __doc__
    """
    _sizeof(arg)
    
        Size in bytes of the underlying MPI handle
    """
    pass

# classes

class Comm(object):
    """ Communicator """
    def Abort(self, errorcode=0): # real signature unknown; restored from __doc__
        """
        Comm.Abort(self, int errorcode=0)
        
                Terminate MPI execution environment
        
                .. warning:: This is a direct call, use it with care!!!.
        """
        pass

    def Allgather(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Allgather(self, sendbuf, recvbuf)
        
                Gather to All, gather data from all processes and
                distribute it to all other processes in a group
        """
        pass

    def allgather(self, sendobj): # real signature unknown; restored from __doc__
        """
        Comm.allgather(self, sendobj)
        Gather to All
        """
        pass

    def Allgatherv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Allgatherv(self, sendbuf, recvbuf)
        
                Gather to All Vector, gather data from all processes and
                distribute it to all other processes in a group providing
                different amount of data and displacements
        """
        pass

    def allreduce(self, sendobj, op=None): # real signature unknown; restored from __doc__
        """
        Comm.allreduce(self, sendobj, op=SUM)
        Reduce to All
        """
        pass

    def Allreduce(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Allreduce(self, sendbuf, recvbuf, Op op=SUM)
        
                All Reduce
        """
        pass

    def Alltoall(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Alltoall(self, sendbuf, recvbuf)
        
                All to All Scatter/Gather, send data from all to all
                processes in a group
        """
        pass

    def alltoall(self, sendobj): # real signature unknown; restored from __doc__
        """
        Comm.alltoall(self, sendobj)
        All to All Scatter/Gather
        """
        pass

    def Alltoallv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Alltoallv(self, sendbuf, recvbuf)
        
                All to All Scatter/Gather Vector, send data from all to all
                processes in a group providing different amount of data and
                displacements
        """
        pass

    def Alltoallw(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Alltoallw(self, sendbuf, recvbuf)
        
                Generalized All-to-All communication allowing different
                counts, displacements and datatypes for each partner
        """
        pass

    def Barrier(self): # real signature unknown; restored from __doc__
        """
        Comm.Barrier(self)
        
                Barrier synchronization
        """
        pass

    def barrier(self): # real signature unknown; restored from __doc__
        """
        Comm.barrier(self)
        Barrier
        """
        pass

    def bcast(self, obj, root=0): # real signature unknown; restored from __doc__
        """
        Comm.bcast(self, obj, int root=0)
        Broadcast
        """
        pass

    def Bcast(self, buf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Bcast(self, buf, int root=0)
        
                Broadcast a message from one process
                to all other processes in a group
        """
        pass

    def bsend(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.bsend(self, obj, int dest, int tag=0)
        Send in buffered mode
        """
        pass

    def Bsend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Bsend(self, buf, int dest, int tag=0)
        
                Blocking send in buffered mode
        """
        pass

    def Bsend_init(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Bsend_init(self, buf, int dest, int tag=0)
        
                Persistent request for a send in buffered mode
        """
        pass

    def Call_errhandler(self, errorcode): # real signature unknown; restored from __doc__
        """
        Comm.Call_errhandler(self, int errorcode)
        
                Call the error handler installed on a communicator
        """
        pass

    def Clone(self): # real signature unknown; restored from __doc__
        """
        Comm.Clone(self)
        
                Clone an existing communicator
        """
        pass

    @classmethod
    def Compare(cls, comm1, comm2): # real signature unknown; restored from __doc__
        """
        Comm.Compare(type cls, Comm comm1, Comm comm2)
        
                Compare two communicators
        """
        pass

    def Create(self, group): # real signature unknown; restored from __doc__
        """
        Comm.Create(self, Group group)
        
                Create communicator from group
        """
        pass

    def Create_group(self, group, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Create_group(self, Group group, int tag=0)
        
                Create communicator from group
        """
        pass

    @classmethod
    def Create_keyval(cls, copy_fn=None, delete_fn=None): # real signature unknown; restored from __doc__
        """
        Comm.Create_keyval(type cls, copy_fn=None, delete_fn=None)
        
                Create a new attribute key for communicators
        """
        pass

    def Delete_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Comm.Delete_attr(self, int keyval)
        
                Delete attribute value associated with a key
        """
        pass

    def Disconnect(self): # real signature unknown; restored from __doc__
        """
        Comm.Disconnect(self)
        
                Disconnect from a communicator
        """
        pass

    def Dup(self, Info_info=None): # real signature unknown; restored from __doc__
        """
        Comm.Dup(self, Info info=None)
        
                Duplicate an existing communicator
        """
        pass

    def Dup_with_info(self, info): # real signature unknown; restored from __doc__
        """
        Comm.Dup_with_info(self, Info info)
        
                Duplicate an existing communicator
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Comm.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Comm.Free(self)
        
                Free a communicator
        """
        pass

    @classmethod
    def Free_keyval(cls, keyval): # real signature unknown; restored from __doc__
        """
        Comm.Free_keyval(type cls, int keyval)
        
                Free and attribute key for communicators
        """
        pass

    def gather(self, sendobj, root=0): # real signature unknown; restored from __doc__
        """
        Comm.gather(self, sendobj, int root=0)
        Gather
        """
        pass

    def Gather(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Gather(self, sendbuf, recvbuf, int root=0)
        
                Gather together values from a group of processes
        """
        pass

    def Gatherv(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Gatherv(self, sendbuf, recvbuf, int root=0)
        
                Gather Vector, gather data to one process from all other
                processes in a group providing different amount of data and
                displacements at the receiving sides
        """
        pass

    def Get_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Comm.Get_attr(self, int keyval)
        
                Retrieve attribute value by key
        """
        pass

    def Get_errhandler(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_errhandler(self)
        
                Get the error handler for a communicator
        """
        pass

    def Get_group(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_group(self)
        
                Access the group associated with a communicator
        """
        pass

    def Get_info(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_info(self)
        
                Return the hints for a communicator
                that are currently in use
        """
        pass

    def Get_name(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_name(self)
        
                Get the print name for this communicator
        """
        pass

    @classmethod
    def Get_parent(cls): # real signature unknown; restored from __doc__
        """
        Comm.Get_parent(type cls)
        
                Return the parent intercommunicator for this process
        """
        pass

    def Get_rank(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_rank(self)
        
                Return the rank of this process in a communicator
        """
        pass

    def Get_size(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_size(self)
        
                Return the number of processes in a communicator
        """
        pass

    def Get_topology(self): # real signature unknown; restored from __doc__
        """
        Comm.Get_topology(self)
        
                Determine the type of topology (if any)
                associated with a communicator
        """
        pass

    def Iallgather(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Iallgather(self, sendbuf, recvbuf)
        
                Nonblocking Gather to All
        """
        pass

    def Iallgatherv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Iallgatherv(self, sendbuf, recvbuf)
        
                Nonblocking Gather to All Vector
        """
        pass

    def Iallreduce(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Iallreduce(self, sendbuf, recvbuf, Op op=SUM)
        
                Nonblocking All Reduce
        """
        pass

    def Ialltoall(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Ialltoall(self, sendbuf, recvbuf)
        
                Nonblocking All to All Scatter/Gather
        """
        pass

    def Ialltoallv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Ialltoallv(self, sendbuf, recvbuf)
        
                Nonblocking All to All Scatter/Gather Vector
        """
        pass

    def Ialltoallw(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Comm.Ialltoallw(self, sendbuf, recvbuf)
        
                Nonblocking Generalized All-to-All
        """
        pass

    def Ibarrier(self): # real signature unknown; restored from __doc__
        """
        Comm.Ibarrier(self)
        
                Nonblocking Barrier
        """
        pass

    def Ibcast(self, buf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Ibcast(self, buf, int root=0)
        
                Nonblocking Broadcast
        """
        pass

    def ibsend(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.ibsend(self, obj, int dest, int tag=0)
        Nonblocking send in buffered mode
        """
        pass

    def Ibsend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Ibsend(self, buf, int dest, int tag=0)
        
                Nonblocking send in buffered mode
        """
        pass

    def Idup(self): # real signature unknown; restored from __doc__
        """
        Comm.Idup(self)
        
                Nonblocking duplicate an existing communicator
        """
        pass

    def Igather(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Igather(self, sendbuf, recvbuf, int root=0)
        
                Nonblocking Gather
        """
        pass

    def Igatherv(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Igatherv(self, sendbuf, recvbuf, int root=0)
        
                Nonblocking Gather Vector
        """
        pass

    def Improbe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Improbe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Nonblocking test for a matched message
        """
        pass

    def improbe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.improbe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Nonblocking test for a matched message
        """
        pass

    def Iprobe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Iprobe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Nonblocking test for a message
        """
        pass

    def iprobe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.iprobe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Nonblocking test for a message
        """
        pass

    def irecv(self, buf=None, source=None, tag=None): # real signature unknown; restored from __doc__
        """
        Comm.irecv(self, buf=None, int source=ANY_SOURCE, int tag=ANY_TAG)
        Nonblocking receive
        """
        pass

    def Irecv(self, buf, source=None, tag=None): # real signature unknown; restored from __doc__
        """
        Comm.Irecv(self, buf, int source=ANY_SOURCE, int tag=ANY_TAG)
        
                Nonblocking receive
        """
        pass

    def Ireduce(self, sendbuf, recvbuf, op=None, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Ireduce(self, sendbuf, recvbuf, Op op=SUM, int root=0)
        
                Nonblocking Reduce
        """
        pass

    def Ireduce_scatter(self, sendbuf, recvbuf, recvcounts=None, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Ireduce_scatter(self, sendbuf, recvbuf, recvcounts=None, Op op=SUM)
        
                Nonblocking Reduce-Scatter (vector version)
        """
        pass

    def Ireduce_scatter_block(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Ireduce_scatter_block(self, sendbuf, recvbuf, Op op=SUM)
        
                Nonblocking Reduce-Scatter Block (regular, non-vector version)
        """
        pass

    def Irsend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Irsend(self, buf, int dest, int tag=0)
        
                Nonblocking send in ready mode
        """
        pass

    def Iscatter(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Iscatter(self, sendbuf, recvbuf, int root=0)
        
                Nonblocking Scatter
        """
        pass

    def Iscatterv(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Iscatterv(self, sendbuf, recvbuf, int root=0)
        
                Nonblocking Scatter Vector
        """
        pass

    def Isend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Isend(self, buf, int dest, int tag=0)
        
                Nonblocking send
        """
        pass

    def isend(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.isend(self, obj, int dest, int tag=0)
        Nonblocking send
        """
        pass

    def Issend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Issend(self, buf, int dest, int tag=0)
        
                Nonblocking send in synchronous mode
        """
        pass

    def issend(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.issend(self, obj, int dest, int tag=0)
        Nonblocking send in synchronous mode
        """
        pass

    def Is_inter(self): # real signature unknown; restored from __doc__
        """
        Comm.Is_inter(self)
        
                Test to see if a comm is an intercommunicator
        """
        pass

    def Is_intra(self): # real signature unknown; restored from __doc__
        """
        Comm.Is_intra(self)
        
                Test to see if a comm is an intracommunicator
        """
        pass

    @classmethod
    def Join(cls, fd): # real signature unknown; restored from __doc__
        """
        Comm.Join(type cls, int fd)
        
                Create a intercommunicator by joining
                two processes connected by a socket
        """
        pass

    def Mprobe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Mprobe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Blocking test for a matched message
        """
        pass

    def mprobe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.mprobe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Blocking test for a matched message
        """
        pass

    def probe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.probe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Blocking test for a message
        """
        pass

    def Probe(self, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Probe(self, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Blocking test for a message
        
                .. note:: This function blocks until the message arrives.
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Comm.py2f(self) """
        pass

    def recv(self, buf=None, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.recv(self, buf=None, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Receive
        """
        pass

    def Recv(self, buf, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Recv(self, buf, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Blocking receive
        
                .. note:: This function blocks until the message is received
        """
        pass

    def Recv_init(self, buf, source=None, tag=None): # real signature unknown; restored from __doc__
        """
        Comm.Recv_init(self, buf, int source=ANY_SOURCE, int tag=ANY_TAG)
        
                Create a persistent request for a receive
        """
        pass

    def reduce(self, sendobj, op=None, root=0): # real signature unknown; restored from __doc__
        """
        Comm.reduce(self, sendobj, op=SUM, int root=0)
        Reduce
        """
        pass

    def Reduce(self, sendbuf, recvbuf, op=None, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Reduce(self, sendbuf, recvbuf, Op op=SUM, int root=0)
        
                Reduce
        """
        pass

    def Reduce_scatter(self, sendbuf, recvbuf, recvcounts=None, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Reduce_scatter(self, sendbuf, recvbuf, recvcounts=None, Op op=SUM)
        
                Reduce-Scatter (vector version)
        """
        pass

    def Reduce_scatter_block(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Comm.Reduce_scatter_block(self, sendbuf, recvbuf, Op op=SUM)
        
                Reduce-Scatter Block (regular, non-vector version)
        """
        pass

    def Rsend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Rsend(self, buf, int dest, int tag=0)
        
                Blocking send in ready mode
        """
        pass

    def Rsend_init(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Rsend_init(self, buf, int dest, int tag=0)
        
                Persistent request for a send in ready mode
        """
        pass

    def scatter(self, sendobj, root=0): # real signature unknown; restored from __doc__
        """
        Comm.scatter(self, sendobj, int root=0)
        Scatter
        """
        pass

    def Scatter(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Scatter(self, sendbuf, recvbuf, int root=0)
        
                Scatter data from one process
                to all other processes in a group
        """
        pass

    def Scatterv(self, sendbuf, recvbuf, root=0): # real signature unknown; restored from __doc__
        """
        Comm.Scatterv(self, sendbuf, recvbuf, int root=0)
        
                Scatter Vector, scatter data from one process to all other
                processes in a group providing different amount of data and
                displacements at the sending side
        """
        pass

    def Send(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Send(self, buf, int dest, int tag=0)
        
                Blocking send
        
                .. note:: This function may block until the message is
                   received. Whether or not `Send` blocks depends on
                   several factors and is implementation dependent
        """
        pass

    def send(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.send(self, obj, int dest, int tag=0)
        Send
        """
        pass

    def Sendrecv(self, sendbuf, dest, sendtag=0, recvbuf=None, source=None, recvtag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Sendrecv(self, sendbuf, int dest, int sendtag=0, recvbuf=None, int source=ANY_SOURCE, int recvtag=ANY_TAG, Status status=None)
        
                Send and receive a message
        
                .. note:: This function is guaranteed not to deadlock in
                   situations where pairs of blocking sends and receives may
                   deadlock.
        
                .. caution:: A common mistake when using this function is to
                   mismatch the tags with the source and destination ranks,
                   which can result in deadlock.
        """
        pass

    def sendrecv(self, sendobj, dest, sendtag=0, recvbuf=None, source=None, recvtag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.sendrecv(self, sendobj, int dest, int sendtag=0, recvbuf=None, int source=ANY_SOURCE, int recvtag=ANY_TAG, Status status=None)
        Send and Receive
        """
        pass

    def Sendrecv_replace(self, buf, dest, sendtag=0, source=None, recvtag=None, status=None): # real signature unknown; restored from __doc__
        """
        Comm.Sendrecv_replace(self, buf, int dest, int sendtag=0, int source=ANY_SOURCE, int recvtag=ANY_TAG, Status status=None)
        
                Send and receive a message
        
                .. note:: This function is guaranteed not to deadlock in
                   situations where pairs of blocking sends and receives may
                   deadlock.
        
                .. caution:: A common mistake when using this function is to
                   mismatch the tags with the source and destination ranks,
                   which can result in deadlock.
        """
        pass

    def Send_init(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Send_init(self, buf, int dest, int tag=0)
        
                Create a persistent request for a standard send
        """
        pass

    def Set_attr(self, keyval, attrval): # real signature unknown; restored from __doc__
        """
        Comm.Set_attr(self, int keyval, attrval)
        
                Store attribute value associated with a key
        """
        pass

    def Set_errhandler(self, Errhandler_errhandler): # real signature unknown; restored from __doc__
        """
        Comm.Set_errhandler(self, Errhandler errhandler)
        
                Set the error handler for a communicator
        """
        pass

    def Set_info(self, Info_info): # real signature unknown; restored from __doc__
        """
        Comm.Set_info(self, Info info)
        
                Set new values for the hints
                associated with a communicator
        """
        pass

    def Set_name(self, name): # real signature unknown; restored from __doc__
        """
        Comm.Set_name(self, name)
        
                Set the print name for this communicator
        """
        pass

    def Split(self, color=0, key=0): # real signature unknown; restored from __doc__
        """
        Comm.Split(self, int color=0, int key=0)
        
                Split communicator by color and key
        """
        pass

    def Split_type(self, split_type, key=0, Info_info=None): # real signature unknown; restored from __doc__
        """
        Comm.Split_type(self, int split_type, int key=0, Info info=INFO_NULL)
        
                Split communicator by color and key
        """
        pass

    def Ssend(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Ssend(self, buf, int dest, int tag=0)
        
                Blocking send in synchronous mode
        """
        pass

    def ssend(self, obj, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.ssend(self, obj, int dest, int tag=0)
        Send in synchronous mode
        """
        pass

    def Ssend_init(self, buf, dest, tag=0): # real signature unknown; restored from __doc__
        """
        Comm.Ssend_init(self, buf, int dest, int tag=0)
        
                Persistent request for a send in synchronous mode
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    group = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """communicator group"""

    info = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """communicator info"""

    is_inter = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is intercommunicator"""

    is_intra = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is intracommunicator"""

    is_topo = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is a topology communicator"""

    name = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """communicator name"""

    rank = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """rank of this process in communicator"""

    size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of processes in communicator"""

    topology = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """communicator topology type"""


    __hash__ = None


class Intracomm(Comm):
    """ Intracommunicator """
    def Accept(self, port_name, Info_info=None, root=0): # real signature unknown; restored from __doc__
        """
        Intracomm.Accept(self, port_name, Info info=INFO_NULL, int root=0)
        
                Accept a request to form a new intercommunicator
        """
        pass

    def Cart_map(self, dims, periods=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Cart_map(self, dims, periods=None)
        
                Return an optimal placement for the
                calling process on the physical machine
        """
        pass

    def Connect(self, port_name, Info_info=None, root=0): # real signature unknown; restored from __doc__
        """
        Intracomm.Connect(self, port_name, Info info=INFO_NULL, int root=0)
        
                Make a request to form a new intercommunicator
        """
        pass

    def Create_cart(self, dims, periods=None, bool_reorder=False): # real signature unknown; restored from __doc__
        """
        Intracomm.Create_cart(self, dims, periods=None, bool reorder=False)
        
                Create cartesian communicator
        """
        pass

    def Create_dist_graph(self, sources, degrees, destinations, weights=None, Info_info=None, bool_reorder=False): # real signature unknown; restored from __doc__
        """
        Intracomm.Create_dist_graph(self, sources, degrees, destinations, weights=None, Info info=INFO_NULL, bool reorder=False)
        
                Create distributed graph communicator
        """
        pass

    def Create_dist_graph_adjacent(self, sources, destinations, sourceweights=None, destweights=None, Info_info=None, bool_reorder=False): # real signature unknown; restored from __doc__
        """
        Intracomm.Create_dist_graph_adjacent(self, sources, destinations, sourceweights=None, destweights=None, Info info=INFO_NULL, bool reorder=False)
        
                Create distributed graph communicator
        """
        pass

    def Create_graph(self, index, edges, bool_reorder=False): # real signature unknown; restored from __doc__
        """
        Intracomm.Create_graph(self, index, edges, bool reorder=False)
        
                Create graph communicator
        """
        pass

    def Create_intercomm(self, local_leader, Intracomm_peer_comm, remote_leader, tag=0): # real signature unknown; restored from __doc__
        """
        Intracomm.Create_intercomm(self, int local_leader, Intracomm peer_comm, int remote_leader, int tag=0)
        
                Create intercommunicator
        """
        pass

    def Exscan(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Exscan(self, sendbuf, recvbuf, Op op=SUM)
        
                Exclusive Scan
        """
        pass

    def exscan(self, sendobj, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.exscan(self, sendobj, op=SUM)
        Exclusive Scan
        """
        pass

    def Graph_map(self, index, edges): # real signature unknown; restored from __doc__
        """
        Intracomm.Graph_map(self, index, edges)
        
                Return an optimal placement for the
                calling process on the physical machine
        """
        pass

    def Iexscan(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Iexscan(self, sendbuf, recvbuf, Op op=SUM)
        
                Inclusive Scan
        """
        pass

    def Iscan(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Iscan(self, sendbuf, recvbuf, Op op=SUM)
        
                Inclusive Scan
        """
        pass

    def scan(self, sendobj, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.scan(self, sendobj, op=SUM)
        Inclusive Scan
        """
        pass

    def Scan(self, sendbuf, recvbuf, op=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Scan(self, sendbuf, recvbuf, Op op=SUM)
        
                Inclusive Scan
        """
        pass

    def Spawn(self, command, args=None, maxprocs=1, Info_info=None, root=0, errcodes=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Spawn(self, command, args=None, int maxprocs=1, Info info=INFO_NULL, int root=0, errcodes=None)
        
                Spawn instances of a single MPI application
        """
        pass

    def Spawn_multiple(self, command, args=None, maxprocs=None, info=None, root=0, errcodes=None): # real signature unknown; restored from __doc__
        """
        Intracomm.Spawn_multiple(self, command, args=None, maxprocs=None, info=INFO_NULL, int root=0, errcodes=None)
        
                Spawn instances of multiple MPI applications
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass


class Topocomm(Intracomm):
    """ Topology intracommunicator """
    def Ineighbor_allgather(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Ineighbor_allgather(self, sendbuf, recvbuf)
        
                Nonblocking Neighbor Gather to All
        """
        pass

    def Ineighbor_allgatherv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Ineighbor_allgatherv(self, sendbuf, recvbuf)
        
                Nonblocking Neighbor Gather to All Vector
        """
        pass

    def Ineighbor_alltoall(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Ineighbor_alltoall(self, sendbuf, recvbuf)
        
                Nonblocking Neighbor All-to-All
        """
        pass

    def Ineighbor_alltoallv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Ineighbor_alltoallv(self, sendbuf, recvbuf)
        
                Nonblocking Neighbor All-to-All Vector
        """
        pass

    def Ineighbor_alltoallw(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Ineighbor_alltoallw(self, sendbuf, recvbuf)
        
                Nonblocking Neighbor All-to-All Generalized
        """
        pass

    def Neighbor_allgather(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Neighbor_allgather(self, sendbuf, recvbuf)
        
                Neighbor Gather to All
        """
        pass

    def neighbor_allgather(self, sendobj): # real signature unknown; restored from __doc__
        """
        Topocomm.neighbor_allgather(self, sendobj)
        Neighbor Gather to All
        """
        pass

    def Neighbor_allgatherv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Neighbor_allgatherv(self, sendbuf, recvbuf)
        
                Neighbor Gather to All Vector
        """
        pass

    def Neighbor_alltoall(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Neighbor_alltoall(self, sendbuf, recvbuf)
        
                Neighbor All-to-All
        """
        pass

    def neighbor_alltoall(self, sendobj): # real signature unknown; restored from __doc__
        """
        Topocomm.neighbor_alltoall(self, sendobj)
        Neighbor All to All Scatter/Gather
        """
        pass

    def Neighbor_alltoallv(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Neighbor_alltoallv(self, sendbuf, recvbuf)
        
                Neighbor All-to-All Vector
        """
        pass

    def Neighbor_alltoallw(self, sendbuf, recvbuf): # real signature unknown; restored from __doc__
        """
        Topocomm.Neighbor_alltoallw(self, sendbuf, recvbuf)
        
                Neighbor All-to-All Generalized
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    degrees = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of incoming and outgoing neighbors"""

    indegree = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of incoming neighbors"""

    inedges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """incoming neighbors"""

    inoutedges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """incoming and outgoing neighbors"""

    outdegree = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of outgoing neighbors"""

    outedges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """outgoing neighbors"""



class Cartcomm(Topocomm):
    """ Cartesian topology intracommunicator """
    def Get_cart_rank(self, coords): # real signature unknown; restored from __doc__
        """
        Cartcomm.Get_cart_rank(self, coords)
        
                Translate logical coordinates to ranks
        """
        pass

    def Get_coords(self, rank): # real signature unknown; restored from __doc__
        """
        Cartcomm.Get_coords(self, int rank)
        
                Translate ranks to logical coordinates
        """
        pass

    def Get_dim(self): # real signature unknown; restored from __doc__
        """
        Cartcomm.Get_dim(self)
        
                Return number of dimensions
        """
        pass

    def Get_topo(self): # real signature unknown; restored from __doc__
        """
        Cartcomm.Get_topo(self)
        
                Return information on the cartesian topology
        """
        pass

    def Shift(self, direction, disp): # real signature unknown; restored from __doc__
        """
        Cartcomm.Shift(self, int direction, int disp)
        
                Return a tuple (source,dest) of process ranks
                for data shifting with Comm.Sendrecv()
        """
        pass

    def Sub(self, remain_dims): # real signature unknown; restored from __doc__
        """
        Cartcomm.Sub(self, remain_dims)
        
                Return cartesian communicators
                that form lower-dimensional subgrids
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    coords = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """coordinates"""

    dim = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of dimensions"""

    dims = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """dimensions"""

    ndim = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of dimensions"""

    periods = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """periodicity"""

    topo = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """topology information"""



class Datatype(object):
    """ Datatype """
    def Commit(self): # real signature unknown; restored from __doc__
        """
        Datatype.Commit(self)
        
                Commit the datatype
        """
        pass

    def Create_contiguous(self, count): # real signature unknown; restored from __doc__
        """
        Datatype.Create_contiguous(self, int count)
        
                Create a contiguous datatype
        """
        pass

    def Create_darray(self, size, rank, gsizes, distribs, dargs, psizes, order=None): # real signature unknown; restored from __doc__
        """
        Datatype.Create_darray(self, int size, int rank, gsizes, distribs, dargs, psizes, int order=ORDER_C)
        
                Create a datatype representing an HPF-like
                distributed array on Cartesian process grids
        """
        pass

    def Create_dup(self, *args, **kwargs): # real signature unknown
        """
        Datatype.Dup(self)
        
                Duplicate a datatype
        """
        pass

    @classmethod
    def Create_f90_complex(cls, p, r): # real signature unknown; restored from __doc__
        """
        Datatype.Create_f90_complex(type cls, int p, int r)
        
                Return a bounded complex datatype
        """
        pass

    @classmethod
    def Create_f90_integer(cls, r): # real signature unknown; restored from __doc__
        """
        Datatype.Create_f90_integer(type cls, int r)
        
                Return a bounded integer datatype
        """
        pass

    @classmethod
    def Create_f90_real(cls, p, r): # real signature unknown; restored from __doc__
        """
        Datatype.Create_f90_real(type cls, int p, int r)
        
                Return a bounded real datatype
        """
        pass

    def Create_hindexed(self, blocklengths, displacements): # real signature unknown; restored from __doc__
        """
        Datatype.Create_hindexed(self, blocklengths, displacements)
        
                Create an indexed datatype
                with displacements in bytes
        """
        pass

    def Create_hindexed_block(self, blocklength, displacements): # real signature unknown; restored from __doc__
        """
        Datatype.Create_hindexed_block(self, int blocklength, displacements)
        
                Create an indexed datatype
                with constant-sized blocks
                and displacements in bytes
        """
        pass

    def Create_hvector(self, count, blocklength, stride): # real signature unknown; restored from __doc__
        """
        Datatype.Create_hvector(self, int count, int blocklength, Aint stride)
        
                Create a vector (strided) datatype
        """
        pass

    def Create_indexed(self, blocklengths, displacements): # real signature unknown; restored from __doc__
        """
        Datatype.Create_indexed(self, blocklengths, displacements)
        
                Create an indexed datatype
        """
        pass

    def Create_indexed_block(self, blocklength, displacements): # real signature unknown; restored from __doc__
        """
        Datatype.Create_indexed_block(self, int blocklength, displacements)
        
                Create an indexed datatype
                with constant-sized blocks
        """
        pass

    @classmethod
    def Create_keyval(cls, copy_fn=None, delete_fn=None): # real signature unknown; restored from __doc__
        """
        Datatype.Create_keyval(type cls, copy_fn=None, delete_fn=None)
        
                Create a new attribute key for datatypes
        """
        pass

    def Create_resized(self, lb, extent): # real signature unknown; restored from __doc__
        """
        Datatype.Create_resized(self, Aint lb, Aint extent)
        
                Create a datatype with a new lower bound and extent
        """
        pass

    @classmethod
    def Create_struct(cls, blocklengths, displacements, datatypes): # real signature unknown; restored from __doc__
        """
        Datatype.Create_struct(type cls, blocklengths, displacements, datatypes)
        
                Create an datatype from a general set of
                block sizes, displacements and datatypes
        """
        pass

    def Create_subarray(self, sizes, subsizes, starts, order=None): # real signature unknown; restored from __doc__
        """
        Datatype.Create_subarray(self, sizes, subsizes, starts, int order=ORDER_C)
        
                Create a datatype for a subarray of
                a regular, multidimensional array
        """
        pass

    def Create_vector(self, count, blocklength, stride): # real signature unknown; restored from __doc__
        """
        Datatype.Create_vector(self, int count, int blocklength, int stride)
        
                Create a vector (strided) datatype
        """
        pass

    def decode(self): # real signature unknown; restored from __doc__
        """
        Datatype.decode(self)
        
                Convenience method for decoding a datatype
        """
        pass

    def Delete_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Datatype.Delete_attr(self, int keyval)
        
                Delete attribute value associated with a key
        """
        pass

    def Dup(self): # real signature unknown; restored from __doc__
        """
        Datatype.Dup(self)
        
                Duplicate a datatype
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Datatype.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Datatype.Free(self)
        
                Free the datatype
        """
        pass

    @classmethod
    def Free_keyval(cls, keyval): # real signature unknown; restored from __doc__
        """
        Datatype.Free_keyval(type cls, int keyval)
        
                Free and attribute key for datatypes
        """
        pass

    def Get_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Datatype.Get_attr(self, int keyval)
        
                Retrieve attribute value by key
        """
        pass

    def Get_contents(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_contents(self)
        
                Retrieve the actual arguments used in the call that created a
                datatype
        """
        pass

    def Get_envelope(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_envelope(self)
        
                Return information on the number and type of input arguments
                used in the call that created a datatype
        """
        pass

    def Get_extent(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_extent(self)
        
                Return lower bound and extent of datatype
        """
        pass

    def Get_name(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_name(self)
        
                Get the print name for this datatype
        """
        pass

    def Get_size(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_size(self)
        
                Return the number of bytes occupied
                by entries in the datatype
        """
        pass

    def Get_true_extent(self): # real signature unknown; restored from __doc__
        """
        Datatype.Get_true_extent(self)
        
                Return the true lower bound and extent of a datatype
        """
        pass

    @classmethod
    def Match_size(cls, typeclass, size): # real signature unknown; restored from __doc__
        """
        Datatype.Match_size(type cls, int typeclass, int size)
        
                Find a datatype matching a specified size in bytes
        """
        pass

    def Pack(self, inbuf, outbuf, position, comm): # real signature unknown; restored from __doc__
        """
        Datatype.Pack(self, inbuf, outbuf, int position, Comm comm)
        
                Pack into contiguous memory according to datatype.
        """
        pass

    def Pack_external(self, datarep, inbuf, outbuf, position): # real signature unknown; restored from __doc__
        """
        Datatype.Pack_external(self, datarep, inbuf, outbuf, Aint position)
        
                Pack into contiguous memory according to datatype,
                using a portable data representation (**external32**).
        """
        pass

    def Pack_external_size(self, datarep, count): # real signature unknown; restored from __doc__
        """
        Datatype.Pack_external_size(self, datarep, int count)
        
                Returns the upper bound on the amount of space (in bytes)
                needed to pack a message according to datatype,
                using a portable data representation (**external32**).
        """
        pass

    def Pack_size(self, count, comm): # real signature unknown; restored from __doc__
        """
        Datatype.Pack_size(self, int count, Comm comm)
        
                Returns the upper bound on the amount of space (in bytes)
                needed to pack a message according to datatype.
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Datatype.py2f(self) """
        pass

    def Resized(self, *args, **kwargs): # real signature unknown
        """
        Datatype.Create_resized(self, Aint lb, Aint extent)
        
                Create a datatype with a new lower bound and extent
        """
        pass

    def Set_attr(self, keyval, attrval): # real signature unknown; restored from __doc__
        """
        Datatype.Set_attr(self, int keyval, attrval)
        
                Store attribute value associated with a key
        """
        pass

    def Set_name(self, name): # real signature unknown; restored from __doc__
        """
        Datatype.Set_name(self, name)
        
                Set the print name for this datatype
        """
        pass

    def Unpack(self, inbuf, position, outbuf, comm): # real signature unknown; restored from __doc__
        """
        Datatype.Unpack(self, inbuf, int position, outbuf, Comm comm)
        
                Unpack from contiguous memory according to datatype.
        """
        pass

    def Unpack_external(self, datarep, inbuf, position, outbuf): # real signature unknown; restored from __doc__
        """
        Datatype.Unpack_external(self, datarep, inbuf, Aint position, outbuf)
        
                Unpack from contiguous memory according to datatype,
                using a portable data representation (**external32**).
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    combiner = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """datatype combiner"""

    contents = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """datatype contents"""

    envelope = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """datatype envelope"""

    extent = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """extent"""

    is_named = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is a named datatype"""

    is_predefined = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is a predefined datatype"""

    lb = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """lower bound"""

    name = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """datatype name"""

    size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """size (in bytes)"""

    true_extent = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """true extent"""

    true_lb = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """true lower bound"""

    true_ub = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """true upper bound"""

    ub = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """upper bound"""


    __hash__ = None


class Distgraphcomm(Topocomm):
    """ Distributed graph topology intracommunicator """
    def Get_dist_neighbors(self): # real signature unknown; restored from __doc__
        """
        Distgraphcomm.Get_dist_neighbors(self)
        
                Return adjacency information for a distributed graph topology
        """
        pass

    def Get_dist_neighbors_count(self): # real signature unknown; restored from __doc__
        """
        Distgraphcomm.Get_dist_neighbors_count(self)
        
                Return adjacency information for a distributed graph topology
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass


class Errhandler(object):
    """ Error Handler """
    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Errhandler.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Errhandler.Free(self)
        
                Free an error handler
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Errhandler.py2f(self) """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    __hash__ = None


class Exception(RuntimeError):
    """ Exception """
    def Get_error_class(self): # real signature unknown; restored from __doc__
        """
        Exception.Get_error_class(self)
        
                Error class
        """
        pass

    def Get_error_code(self): # real signature unknown; restored from __doc__
        """
        Exception.Get_error_code(self)
        
                Error code
        """
        pass

    def Get_error_string(self): # real signature unknown; restored from __doc__
        """
        Exception.Get_error_string(self)
        
                Error string
        """
        pass

    def __bool__(self): # real signature unknown; restored from __doc__
        """ Exception.__bool__(self) """
        pass

    def __eq__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__eq__(self, error) """
        pass

    def __ge__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__ge__(self, error) """
        pass

    def __gt__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__gt__(self, error) """
        pass

    def __hash__(self): # real signature unknown; restored from __doc__
        """ Exception.__hash__(self) """
        pass

    def __init__(self, ierr=0): # real signature unknown; restored from __doc__
        """ Exception.__init__(self, int ierr=0) """
        pass

    def ___(self): # real signature unknown; restored from __doc__
        """ Exception.___(self) """
        pass

    def __le__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__le__(self, error) """
        pass

    def __lt__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__lt__(self, error) """
        pass

    def __ne__(self, error): # real signature unknown; restored from __doc__
        """ Exception.__ne__(self, error) """
        pass

    def __nonzero__(self, *args, **kwargs): # real signature unknown
        """ Exception.__bool__(self) """
        pass

    def __repr__(self): # real signature unknown; restored from __doc__
        """ Exception.__repr__(self) """
        pass

    def __str__(self): # real signature unknown; restored from __doc__
        """ Exception.__str__(self) """
        pass

    error_class = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """error class"""

    error_code = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """error code"""

    error_string = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """error string"""

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""



class File(object):
    """ File """
    def Call_errhandler(self, errorcode): # real signature unknown; restored from __doc__
        """
        File.Call_errhandler(self, int errorcode)
        
                Call the error handler installed on a file
        """
        pass

    def Close(self): # real signature unknown; restored from __doc__
        """
        File.Close(self)
        
                Close a file
        """
        pass

    @classmethod
    def Delete(cls, filename, Info_info=None): # real signature unknown; restored from __doc__
        """
        File.Delete(type cls, filename, Info info=INFO_NULL)
        
                Delete a file
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ File.f2py(type cls, arg) """
        pass

    def Get_amode(self): # real signature unknown; restored from __doc__
        """
        File.Get_amode(self)
        
                Return the file access mode
        """
        pass

    def Get_atomicity(self): # real signature unknown; restored from __doc__
        """
        File.Get_atomicity(self)
        
                Return the atomicity mode
        """
        pass

    def Get_byte_offset(self, Offset_offset): # real signature unknown; restored from __doc__
        """
        File.Get_byte_offset(self, Offset offset)
        
                Returns the absolute byte position in the file corresponding
                to 'offset' etypes relative to the current view
        """
        pass

    def Get_errhandler(self): # real signature unknown; restored from __doc__
        """
        File.Get_errhandler(self)
        
                Get the error handler for a file
        """
        pass

    def Get_group(self): # real signature unknown; restored from __doc__
        """
        File.Get_group(self)
        
                Return the group of processes
                that opened the file
        """
        pass

    def Get_info(self): # real signature unknown; restored from __doc__
        """
        File.Get_info(self)
        
                Return the hints for a file that
                that are currently in use
        """
        pass

    def Get_position(self): # real signature unknown; restored from __doc__
        """
        File.Get_position(self)
        
                Return the current position of the individual file pointer
                in etype units relative to the current view
        """
        pass

    def Get_position_shared(self): # real signature unknown; restored from __doc__
        """
        File.Get_position_shared(self)
        
                Return the current position of the shared file pointer
                in etype units relative to the current view
        """
        pass

    def Get_size(self): # real signature unknown; restored from __doc__
        """
        File.Get_size(self)
        
                Return the file size
        """
        pass

    def Get_type_extent(self, Datatype_datatype): # real signature unknown; restored from __doc__
        """
        File.Get_type_extent(self, Datatype datatype)
        
                Return the extent of datatype in the file
        """
        pass

    def Get_view(self): # real signature unknown; restored from __doc__
        """
        File.Get_view(self)
        
                Return the file view
        """
        pass

    def Iread(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iread(self, buf)
        
                Nonblocking read using individual file pointer
        """
        pass

    def Iread_all(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iread_all(self, buf)
        
                Nonblocking collective read using individual file pointer
        """
        pass

    def Iread_at(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Iread_at(self, Offset offset, buf)
        
                Nonblocking read using explicit offset
        """
        pass

    def Iread_at_all(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Iread_at_all(self, Offset offset, buf)
        
                Nonblocking collective read using explicit offset
        """
        pass

    def Iread_shared(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iread_shared(self, buf)
        
                Nonblocking read using shared file pointer
        """
        pass

    def Iwrite(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iwrite(self, buf)
        
                Nonblocking write using individual file pointer
        """
        pass

    def Iwrite_all(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iwrite_all(self, buf)
        
                Nonblocking collective write using individual file pointer
        """
        pass

    def Iwrite_at(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Iwrite_at(self, Offset offset, buf)
        
                Nonblocking write using explicit offset
        """
        pass

    def Iwrite_at_all(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Iwrite_at_all(self, Offset offset, buf)
        
                Nonblocking collective write using explicit offset
        """
        pass

    def Iwrite_shared(self, buf): # real signature unknown; restored from __doc__
        """
        File.Iwrite_shared(self, buf)
        
                Nonblocking write using shared file pointer
        """
        pass

    @classmethod
    def Open(cls, Intracomm_comm, filename, amode=None, Info_info=None): # real signature unknown; restored from __doc__
        """
        File.Open(type cls, Intracomm comm, filename, int amode=MODE_RDONLY, Info info=INFO_NULL)
        
                Open a file
        """
        pass

    def Preallocate(self, Offset_size): # real signature unknown; restored from __doc__
        """
        File.Preallocate(self, Offset size)
        
                Preallocate storage space for a file
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ File.py2f(self) """
        pass

    def Read(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read(self, buf, Status status=None)
        
                Read using individual file pointer
        """
        pass

    def Read_all(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_all(self, buf, Status status=None)
        
                Collective read using individual file pointer
        """
        pass

    def Read_all_begin(self, buf): # real signature unknown; restored from __doc__
        """
        File.Read_all_begin(self, buf)
        
                Start a split collective read
                using individual file pointer
        """
        pass

    def Read_all_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_all_end(self, buf, Status status=None)
        
                Complete a split collective read
                using individual file pointer
        """
        pass

    def Read_at(self, Offset_offset, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_at(self, Offset offset, buf, Status status=None)
        
                Read using explicit offset
        """
        pass

    def Read_at_all(self, Offset_offset, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_at_all(self, Offset offset, buf, Status status=None)
        
                Collective read using explicit offset
        """
        pass

    def Read_at_all_begin(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Read_at_all_begin(self, Offset offset, buf)
        
                Start a split collective read using explict offset
        """
        pass

    def Read_at_all_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_at_all_end(self, buf, Status status=None)
        
                Complete a split collective read using explict offset
        """
        pass

    def Read_ordered(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_ordered(self, buf, Status status=None)
        
                Collective read using shared file pointer
        """
        pass

    def Read_ordered_begin(self, buf): # real signature unknown; restored from __doc__
        """
        File.Read_ordered_begin(self, buf)
        
                Start a split collective read
                using shared file pointer
        """
        pass

    def Read_ordered_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_ordered_end(self, buf, Status status=None)
        
                Complete a split collective read
                using shared file pointer
        """
        pass

    def Read_shared(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Read_shared(self, buf, Status status=None)
        
                Read using shared file pointer
        """
        pass

    def Seek(self, Offset_offset, whence=None): # real signature unknown; restored from __doc__
        """
        File.Seek(self, Offset offset, int whence=SEEK_SET)
        
                Update the individual file pointer
        """
        pass

    def Seek_shared(self, Offset_offset, whence=None): # real signature unknown; restored from __doc__
        """
        File.Seek_shared(self, Offset offset, int whence=SEEK_SET)
        
                Update the shared file pointer
        """
        pass

    def Set_atomicity(self, bool_flag): # real signature unknown; restored from __doc__
        """
        File.Set_atomicity(self, bool flag)
        
                Set the atomicity mode
        """
        pass

    def Set_errhandler(self, Errhandler_errhandler): # real signature unknown; restored from __doc__
        """
        File.Set_errhandler(self, Errhandler errhandler)
        
                Set the error handler for a file
        """
        pass

    def Set_info(self, Info_info): # real signature unknown; restored from __doc__
        """
        File.Set_info(self, Info info)
        
                Set new values for the hints
                associated with a file
        """
        pass

    def Set_size(self, Offset_size): # real signature unknown; restored from __doc__
        """
        File.Set_size(self, Offset size)
        
                Sets the file size
        """
        pass

    def Set_view(self, Offset_disp=0, Datatype_etype=None, Datatype_filetype=None, datarep=None, Info_info=None): # real signature unknown; restored from __doc__
        """
        File.Set_view(self, Offset disp=0, Datatype etype=None, Datatype filetype=None, datarep=None, Info info=INFO_NULL)
        
                Set the file view
        """
        pass

    def Sync(self): # real signature unknown; restored from __doc__
        """
        File.Sync(self)
        
                Causes all previous writes to be
                transferred to the storage device
        """
        pass

    def Write(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write(self, buf, Status status=None)
        
                Write using individual file pointer
        """
        pass

    def Write_all(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_all(self, buf, Status status=None)
        
                Collective write using individual file pointer
        """
        pass

    def Write_all_begin(self, buf): # real signature unknown; restored from __doc__
        """
        File.Write_all_begin(self, buf)
        
                Start a split collective write
                using individual file pointer
        """
        pass

    def Write_all_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_all_end(self, buf, Status status=None)
        
                Complete a split collective write
                using individual file pointer
        """
        pass

    def Write_at(self, Offset_offset, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_at(self, Offset offset, buf, Status status=None)
        
                Write using explicit offset
        """
        pass

    def Write_at_all(self, Offset_offset, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_at_all(self, Offset offset, buf, Status status=None)
        
                Collective write using explicit offset
        """
        pass

    def Write_at_all_begin(self, Offset_offset, buf): # real signature unknown; restored from __doc__
        """
        File.Write_at_all_begin(self, Offset offset, buf)
        
                Start a split collective write using explict offset
        """
        pass

    def Write_at_all_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_at_all_end(self, buf, Status status=None)
        
                Complete a split collective write using explict offset
        """
        pass

    def Write_ordered(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_ordered(self, buf, Status status=None)
        
                Collective write using shared file pointer
        """
        pass

    def Write_ordered_begin(self, buf): # real signature unknown; restored from __doc__
        """
        File.Write_ordered_begin(self, buf)
        
                Start a split collective write using
                shared file pointer
        """
        pass

    def Write_ordered_end(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_ordered_end(self, buf, Status status=None)
        
                Complete a split collective write
                using shared file pointer
        """
        pass

    def Write_shared(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        File.Write_shared(self, buf, Status status=None)
        
                Write using shared file pointer
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    amode = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """file access mode"""

    atomicity = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """atomicity"""

    group = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """file group"""

    info = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """file info"""

    size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """file size"""


    __hash__ = None


class Graphcomm(Topocomm):
    """ General graph topology intracommunicator """
    def Get_dims(self): # real signature unknown; restored from __doc__
        """
        Graphcomm.Get_dims(self)
        
                Return the number of nodes and edges
        """
        pass

    def Get_neighbors(self, rank): # real signature unknown; restored from __doc__
        """
        Graphcomm.Get_neighbors(self, int rank)
        
                Return list of neighbors of a process
        """
        pass

    def Get_neighbors_count(self, rank): # real signature unknown; restored from __doc__
        """
        Graphcomm.Get_neighbors_count(self, int rank)
        
                Return number of neighbors of a process
        """
        pass

    def Get_topo(self): # real signature unknown; restored from __doc__
        """
        Graphcomm.Get_topo(self)
        
                Return index and edges
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    dims = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of nodes and edges"""

    edges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """edges"""

    index = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """index"""

    nedges = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of edges"""

    neighbors = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """neighbors"""

    nneighbors = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of neighbors"""

    nnodes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of nodes"""

    topo = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """topology information"""



class Request(object):
    """ Request """
    def Cancel(self): # real signature unknown; restored from __doc__
        """
        Request.Cancel(self)
        
                Cancel a communication request
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Request.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Request.Free(self)
        
                Free a communication request
        """
        pass

    def Get_status(self, status=None): # real signature unknown; restored from __doc__
        """
        Request.Get_status(self, Status status=None)
        
                Non-destructive test for the completion of a request
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Request.py2f(self) """
        pass

    def test(self, status=None): # real signature unknown; restored from __doc__
        """
        Request.test(self, Status status=None)
        
                Test for the completion of a send or receive
        """
        pass

    def Test(self, status=None): # real signature unknown; restored from __doc__
        """
        Request.Test(self, Status status=None)
        
                Test for the completion of a send or receive
        """
        pass

    @classmethod
    def Testall(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.Testall(type cls, requests, statuses=None)
        
                Test for completion of all previously initiated requests
        """
        pass

    @classmethod
    def testall(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.testall(type cls, requests, statuses=None)
        
                Test for completion of all previously initiated requests
        """
        pass

    @classmethod
    def Testany(cls, requests, status=None): # real signature unknown; restored from __doc__
        """
        Request.Testany(type cls, requests, Status status=None)
        
                Test for completion of any previously initiated request
        """
        pass

    @classmethod
    def testany(cls, requests, status=None): # real signature unknown; restored from __doc__
        """
        Request.testany(type cls, requests, Status status=None)
        
                Test for completion of any previously initiated request
        """
        pass

    @classmethod
    def Testsome(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.Testsome(type cls, requests, statuses=None)
        
                Test for completion of some previously initiated requests
        """
        pass

    def Wait(self, status=None): # real signature unknown; restored from __doc__
        """
        Request.Wait(self, Status status=None)
        
                Wait for a send or receive to complete
        """
        pass

    def wait(self, status=None): # real signature unknown; restored from __doc__
        """
        Request.wait(self, Status status=None)
        
                Wait for a send or receive to complete
        """
        pass

    @classmethod
    def Waitall(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.Waitall(type cls, requests, statuses=None)
        
                Wait for all previously initiated requests to complete
        """
        pass

    @classmethod
    def waitall(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.waitall(type cls, requests, statuses=None)
        
                Wait for all previously initiated requests to complete
        """
        pass

    @classmethod
    def waitany(cls, requests, status=None): # real signature unknown; restored from __doc__
        """
        Request.waitany(type cls, requests, Status status=None)
        
                Wait for any previously initiated request to complete
        """
        pass

    @classmethod
    def Waitany(cls, requests, status=None): # real signature unknown; restored from __doc__
        """
        Request.Waitany(type cls, requests, Status status=None)
        
                Wait for any previously initiated request to complete
        """
        pass

    @classmethod
    def Waitsome(cls, requests, statuses=None): # real signature unknown; restored from __doc__
        """
        Request.Waitsome(type cls, requests, statuses=None)
        
                Wait for some previously initiated requests to complete
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    __hash__ = None


class Grequest(Request):
    """ Generalized request """
    def Complete(self): # real signature unknown; restored from __doc__
        """
        Grequest.Complete(self)
        
                Notify that a user-defined request is complete
        """
        pass

    @classmethod
    def Start(cls, query_fn, free_fn, cancel_fn, args=None, kargs=None): # real signature unknown; restored from __doc__
        """
        Grequest.Start(type cls, query_fn, free_fn, cancel_fn, args=None, kargs=None)
        
                Create and return a user-defined request
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass


class Group(object):
    """ Group """
    @classmethod
    def Compare(cls, group1, group2): # real signature unknown; restored from __doc__
        """
        Group.Compare(type cls, Group group1, Group group2)
        
                Compare two groups
        """
        pass

    @classmethod
    def Difference(cls, group1, group2): # real signature unknown; restored from __doc__
        """
        Group.Difference(type cls, Group group1, Group group2)
        
                Produce a group from the difference
                of two existing groups
        """
        pass

    def Dup(self): # real signature unknown; restored from __doc__
        """
        Group.Dup(self)
        
                Duplicate a group
        """
        pass

    def Excl(self, ranks): # real signature unknown; restored from __doc__
        """
        Group.Excl(self, ranks)
        
                Produce a group by reordering an existing
                group and taking only unlisted members
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Group.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Group.Free(self)
        
                Free a group
        """
        pass

    def Get_rank(self): # real signature unknown; restored from __doc__
        """
        Group.Get_rank(self)
        
                Return the rank of this process in a group
        """
        pass

    def Get_size(self): # real signature unknown; restored from __doc__
        """
        Group.Get_size(self)
        
                Return the size of a group
        """
        pass

    def Incl(self, ranks): # real signature unknown; restored from __doc__
        """
        Group.Incl(self, ranks)
        
                Produce a group by reordering an existing
                group and taking only listed members
        """
        pass

    def Intersect(self, *args, **kwargs): # real signature unknown
        """
        Group.Intersection(type cls, Group group1, Group group2)
        
                Produce a group as the intersection
                of two existing groups
        """
        pass

    @classmethod
    def Intersection(cls, group1, group2): # real signature unknown; restored from __doc__
        """
        Group.Intersection(type cls, Group group1, Group group2)
        
                Produce a group as the intersection
                of two existing groups
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Group.py2f(self) """
        pass

    def Range_excl(self, ranks): # real signature unknown; restored from __doc__
        """
        Group.Range_excl(self, ranks)
        
                Create a new group by excluding ranges
                of processes from an existing group
        """
        pass

    def Range_incl(self, ranks): # real signature unknown; restored from __doc__
        """
        Group.Range_incl(self, ranks)
        
                Create a new group from ranges of
                of ranks in an existing group
        """
        pass

    @classmethod
    def Translate_ranks(cls, group1, ranks1, group2=None): # real signature unknown; restored from __doc__
        """
        Group.Translate_ranks(type cls, Group group1, ranks1, Group group2=None)
        
                Translate the ranks of processes in
                one group to those in another group
        """
        pass

    @classmethod
    def Union(cls, group1, group2): # real signature unknown; restored from __doc__
        """
        Group.Union(type cls, Group group1, Group group2)
        
                Produce a group by combining
                two existing groups
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    rank = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """rank of this process in group"""

    size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of processes in group"""


    __hash__ = None


class Info(object):
    """ Info """
    def clear(self): # real signature unknown; restored from __doc__
        """
        Info.clear(self)
        info clear
        """
        pass

    @classmethod
    def Create(cls): # real signature unknown; restored from __doc__
        """
        Info.Create(type cls)
        
                Create a new, empty info object
        """
        pass

    def Delete(self, key): # real signature unknown; restored from __doc__
        """
        Info.Delete(self, key)
        
                Remove a (key, value) pair from info
        """
        pass

    def Dup(self): # real signature unknown; restored from __doc__
        """
        Info.Dup(self)
        
                Duplicate an existing info object, creating a new object, with
                the same (key, value) pairs and the same ordering of keys
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Info.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Info.Free(self)
        
                Free a info object
        """
        pass

    def Get(self, key, maxlen=-1): # real signature unknown; restored from __doc__
        """
        Info.Get(self, key, int maxlen=-1)
        
                Retrieve the value associated with a key
        """
        pass

    def get(self, key, default=None): # real signature unknown; restored from __doc__
        """
        Info.get(self, key, default=None)
        info get
        """
        pass

    def Get_nkeys(self): # real signature unknown; restored from __doc__
        """
        Info.Get_nkeys(self)
        
                Return the number of currently defined keys in info
        """
        pass

    def Get_nthkey(self, n): # real signature unknown; restored from __doc__
        """
        Info.Get_nthkey(self, int n)
        
                Return the nth defined key in info. Keys are numbered in the
                range [0, N) where N is the value returned by
                `Info.Get_nkeys()`
        """
        pass

    def items(self): # real signature unknown; restored from __doc__
        """
        Info.items(self)
        info items
        """
        pass

    def keys(self): # real signature unknown; restored from __doc__
        """
        Info.keys(self)
        info keys
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Info.py2f(self) """
        pass

    def Set(self, key, value): # real signature unknown; restored from __doc__
        """
        Info.Set(self, key, value)
        
                Add the (key, value) pair to info, and overrides the value if
                a value for the same key was previously set
        """
        pass

    def update(self, other=(), **kwds): # real signature unknown; restored from __doc__
        """
        Info.update(self, other=(), **kwds)
        info update
        """
        pass

    def values(self): # real signature unknown; restored from __doc__
        """
        Info.values(self)
        info values
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __contains__(self, *args, **kwargs): # real signature unknown
        """ Return key in self. """
        pass

    def __delitem__(self, *args, **kwargs): # real signature unknown
        """ Delete self[key]. """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __getitem__(self, *args, **kwargs): # real signature unknown
        """ Return self[key]. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __iter__(self, *args, **kwargs): # real signature unknown
        """ Implement iter(self). """
        pass

    def __len__(self, *args, **kwargs): # real signature unknown
        """ Return len(self). """
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    def __setitem__(self, *args, **kwargs): # real signature unknown
        """ Set self[key] to value. """
        pass

    __hash__ = None


class Intercomm(Comm):
    """ Intercommunicator """
    def Get_remote_group(self): # real signature unknown; restored from __doc__
        """
        Intercomm.Get_remote_group(self)
        
                Access the remote group associated
                with the inter-communicator
        """
        pass

    def Get_remote_size(self): # real signature unknown; restored from __doc__
        """
        Intercomm.Get_remote_size(self)
        
                Intercommunicator remote size
        """
        pass

    def Merge(self, bool_high=False): # real signature unknown; restored from __doc__
        """
        Intercomm.Merge(self, bool high=False)
        
                Merge intercommunicator
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    remote_group = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """remote group"""

    remote_size = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """number of remote processes"""



class Message(object):
    """ Message """
    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Message.f2py(type cls, arg) """
        pass

    @classmethod
    def Iprobe(cls, comm, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Message.Iprobe(type cls, Comm comm, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Nonblocking test for a matched message
        """
        pass

    @classmethod
    def iprobe(cls, comm, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Message.iprobe(type cls, Comm comm, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Nonblocking test for a matched message
        """
        pass

    def irecv(self): # real signature unknown; restored from __doc__
        """
        Message.irecv(self)
        Nonblocking receive of matched message
        """
        pass

    def Irecv(self, buf): # real signature unknown; restored from __doc__
        """
        Message.Irecv(self, buf)
        
                Nonblocking receive of matched message
        """
        pass

    @classmethod
    def probe(cls, comm, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Message.probe(type cls, Comm comm, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        Blocking test for a matched message
        """
        pass

    @classmethod
    def Probe(cls, comm, source=None, tag=None, status=None): # real signature unknown; restored from __doc__
        """
        Message.Probe(type cls, Comm comm, int source=ANY_SOURCE, int tag=ANY_TAG, Status status=None)
        
                Blocking test for a matched message
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Message.py2f(self) """
        pass

    def Recv(self, buf, status=None): # real signature unknown; restored from __doc__
        """
        Message.Recv(self, buf, Status status=None)
        
                Blocking receive of matched message
        """
        pass

    def recv(self, status=None): # real signature unknown; restored from __doc__
        """
        Message.recv(self, Status status=None)
        Blocking receive of matched message
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    __hash__ = None


class Op(object):
    """ Op """
    @classmethod
    def Create(cls, function, bool_commute=False): # real signature unknown; restored from __doc__
        """
        Op.Create(type cls, function, bool commute=False)
        
                Create a user-defined operation
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Op.f2py(type cls, arg) """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Op.Free(self)
        
                Free the operation
        """
        pass

    def Is_commutative(self): # real signature unknown; restored from __doc__
        """
        Op.Is_commutative(self)
        
                Query reduction operations for their commutativity
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Op.py2f(self) """
        pass

    def Reduce_local(self, inbuf, inoutbuf): # real signature unknown; restored from __doc__
        """
        Op.Reduce_local(self, inbuf, inoutbuf)
        
                Apply a reduction operator to local data
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __call__(self, *args, **kwargs): # real signature unknown
        """ Call self as a function. """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    is_commutative = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is commutative"""

    is_predefined = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """is a predefined operation"""


    __hash__ = None


class Prequest(Request):
    """ Persistent request """
    def Start(self): # real signature unknown; restored from __doc__
        """
        Prequest.Start(self)
        
                Initiate a communication with a persistent request
        """
        pass

    @classmethod
    def Startall(cls, requests): # real signature unknown; restored from __doc__
        """
        Prequest.Startall(type cls, requests)
        
                Start a collection of persistent requests
        """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass


class Status(object):
    """ Status """
    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Status.f2py(type cls, arg) """
        pass

    def Get_count(self, Datatype_datatype=None): # real signature unknown; restored from __doc__
        """
        Status.Get_count(self, Datatype datatype=BYTE)
        
                Get the number of *top level* elements
        """
        pass

    def Get_elements(self, Datatype_datatype): # real signature unknown; restored from __doc__
        """
        Status.Get_elements(self, Datatype datatype)
        
                Get the number of basic elements in a datatype
        """
        pass

    def Get_error(self): # real signature unknown; restored from __doc__
        """
        Status.Get_error(self)
        
                Get message error
        """
        pass

    def Get_source(self): # real signature unknown; restored from __doc__
        """
        Status.Get_source(self)
        
                Get message source
        """
        pass

    def Get_tag(self): # real signature unknown; restored from __doc__
        """
        Status.Get_tag(self)
        
                Get message tag
        """
        pass

    def Is_cancelled(self): # real signature unknown; restored from __doc__
        """
        Status.Is_cancelled(self)
        
                Test to see if a request was cancelled
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Status.py2f(self) """
        pass

    def Set_cancelled(self, bool_flag): # real signature unknown; restored from __doc__
        """
        Status.Set_cancelled(self, bool flag)
        
                Set the cancelled state associated with a status
        
                .. note:: This should be only used when implementing
                   query callback functions for generalized requests
        """
        pass

    def Set_elements(self, Datatype_datatype, Count_count): # real signature unknown; restored from __doc__
        """
        Status.Set_elements(self, Datatype datatype, Count count)
        
                Set the number of elements in a status
        
                .. note:: This should be only used when implementing
                   query callback functions for generalized requests
        """
        pass

    def Set_error(self, error): # real signature unknown; restored from __doc__
        """
        Status.Set_error(self, int error)
        
                Set message error
        """
        pass

    def Set_source(self, source): # real signature unknown; restored from __doc__
        """
        Status.Set_source(self, int source)
        
                Set message source
        """
        pass

    def Set_tag(self, tag): # real signature unknown; restored from __doc__
        """
        Status.Set_tag(self, int tag)
        
                Set message tag
        """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    cancelled = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """
        cancelled state
        """

    count = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """byte count"""

    error = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """error"""

    source = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """source"""

    tag = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """tag"""


    __hash__ = None


class Win(object):
    """ Window """
    def Accumulate(self, origin, target_rank, target=None, op=None): # real signature unknown; restored from __doc__
        """
        Win.Accumulate(self, origin, int target_rank, target=None, Op op=SUM)
        
                Accumulate data into the target process
        """
        pass

    @classmethod
    def Allocate(cls, size, disp_unit=1, Info_info=None, Intracomm_comm=None): # real signature unknown; restored from __doc__
        """
        Win.Allocate(type cls, Aint size, int disp_unit=1, Info info=INFO_NULL, Intracomm comm=COMM_SELF)
        
                Create an window object for one-sided communication
        """
        pass

    @classmethod
    def Allocate_shared(cls, size, disp_unit=1, Info_info=None, Intracomm_comm=None): # real signature unknown; restored from __doc__
        """
        Win.Allocate_shared(type cls, Aint size, int disp_unit=1, Info info=INFO_NULL, Intracomm comm=COMM_SELF)
        
                Create an window object for one-sided communication
        """
        pass

    def Attach(self, memory): # real signature unknown; restored from __doc__
        """
        Win.Attach(self, memory)
        
                Attach a local memory region
        """
        pass

    def Call_errhandler(self, errorcode): # real signature unknown; restored from __doc__
        """
        Win.Call_errhandler(self, int errorcode)
        
                Call the error handler installed on a window
        """
        pass

    def Complete(self): # real signature unknown; restored from __doc__
        """
        Win.Complete(self)
        
                Completes an RMA operations begun after an `Win.Start()`
        """
        pass

    @classmethod
    def Create(cls, memory, disp_unit=1, Info_info=None, Intracomm_comm=None): # real signature unknown; restored from __doc__
        """
        Win.Create(type cls, memory, int disp_unit=1, Info info=INFO_NULL, Intracomm comm=COMM_SELF)
        
                Create an window object for one-sided communication
        """
        pass

    @classmethod
    def Create_dynamic(cls, Info_info=None, Intracomm_comm=None): # real signature unknown; restored from __doc__
        """
        Win.Create_dynamic(type cls, Info info=INFO_NULL, Intracomm comm=COMM_SELF)
        
                Create an window object for one-sided communication
        """
        pass

    @classmethod
    def Create_keyval(cls, copy_fn=None, delete_fn=None): # real signature unknown; restored from __doc__
        """
        Win.Create_keyval(type cls, copy_fn=None, delete_fn=None)
        
                Create a new attribute key for windows
        """
        pass

    def Delete_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Win.Delete_attr(self, int keyval)
        
                Delete attribute value associated with a key
        """
        pass

    def Detach(self, memory): # real signature unknown; restored from __doc__
        """
        Win.Detach(self, memory)
        
                Detach a local memory region
        """
        pass

    @classmethod
    def f2py(cls, arg): # real signature unknown; restored from __doc__
        """ Win.f2py(type cls, arg) """
        pass

    def Fence(self, assertion=0): # real signature unknown; restored from __doc__
        """
        Win.Fence(self, int assertion=0)
        
                Perform an MPI fence synchronization on a window
        """
        pass

    def Flush(self, rank): # real signature unknown; restored from __doc__
        """
        Win.Flush(self, int rank)
        
                Complete all outstanding RMA operations at the given target
        """
        pass

    def Flush_all(self): # real signature unknown; restored from __doc__
        """
        Win.Flush_all(self)
        
                Complete  all  outstanding RMA operations at all targets
        """
        pass

    def Flush_local(self, rank): # real signature unknown; restored from __doc__
        """
        Win.Flush_local(self, int rank)
        
                Complete locally all outstanding RMA operations at the given target
        """
        pass

    def Flush_local_all(self): # real signature unknown; restored from __doc__
        """
        Win.Flush_local_all(self)
        
                Complete locally all outstanding RMA opera- tions at all targets
        """
        pass

    def Free(self): # real signature unknown; restored from __doc__
        """
        Win.Free(self)
        
                Free a window
        """
        pass

    @classmethod
    def Free_keyval(cls, keyval): # real signature unknown; restored from __doc__
        """
        Win.Free_keyval(type cls, int keyval)
        
                Free and attribute key for windows
        """
        pass

    def Get(self, origin, target_rank, target=None): # real signature unknown; restored from __doc__
        """
        Win.Get(self, origin, int target_rank, target=None)
        
                Get data from a memory window on a remote process.
        """
        pass

    def Get_accumulate(self, origin, result, target_rank, target=None, op=None): # real signature unknown; restored from __doc__
        """
        Win.Get_accumulate(self, origin, result, int target_rank, target=None, Op op=SUM)
        
                Fetch-and-accumulate data into the target process
        """
        pass

    def Get_attr(self, keyval): # real signature unknown; restored from __doc__
        """
        Win.Get_attr(self, int keyval)
        
                Retrieve attribute value by key
        """
        pass

    def Get_errhandler(self): # real signature unknown; restored from __doc__
        """
        Win.Get_errhandler(self)
        
                Get the error handler for a window
        """
        pass

    def Get_group(self): # real signature unknown; restored from __doc__
        """
        Win.Get_group(self)
        
                Return a duplicate of the group of the
                communicator used to create the window
        """
        pass

    def Get_info(self): # real signature unknown; restored from __doc__
        """
        Win.Get_info(self)
        
                Return the hints for a windows
                that are currently in use
        """
        pass

    def Get_name(self): # real signature unknown; restored from __doc__
        """
        Win.Get_name(self)
        
                Get the print name associated with the window
        """
        pass

    def Lock(self, rank, lock_type=None, assertion=0): # real signature unknown; restored from __doc__
        """
        Win.Lock(self, int rank, int lock_type=LOCK_EXCLUSIVE, int assertion=0)
        
                Begin an RMA access epoch at the target process
        """
        pass

    def Lock_all(self, assertion=0): # real signature unknown; restored from __doc__
        """
        Win.Lock_all(self, int assertion=0)
        
                Begin an RMA access epoch at all processes
        """
        pass

    def Post(self, group, assertion=0): # real signature unknown; restored from __doc__
        """
        Win.Post(self, Group group, int assertion=0)
        
                Start an RMA exposure epoch
        """
        pass

    def Put(self, origin, target_rank, target=None): # real signature unknown; restored from __doc__
        """
        Win.Put(self, origin, int target_rank, target=None)
        
                Put data into a memory window on a remote process.
        """
        pass

    def py2f(self): # real signature unknown; restored from __doc__
        """ Win.py2f(self) """
        pass

    def Raccumulate(self, origin, target_rank, target=None, op=None): # real signature unknown; restored from __doc__
        """
        Win.Raccumulate(self, origin, int target_rank, target=None, Op op=SUM)
        
                Fetch-and-accumulate data into the target process
        """
        pass

    def Rget(self, origin, target_rank, target=None): # real signature unknown; restored from __doc__
        """
        Win.Rget(self, origin, int target_rank, target=None)
        
                Get data from a memory window on a remote process.
        """
        pass

    def Rget_accumulate(self, origin, result, target_rank, target=None, op=None): # real signature unknown; restored from __doc__
        """
        Win.Rget_accumulate(self, origin, result, int target_rank, target=None, Op op=SUM)
        
                Accumulate data into the target process
                using remote memory access.
        """
        pass

    def Rput(self, origin, target_rank, target=None): # real signature unknown; restored from __doc__
        """
        Win.Rput(self, origin, int target_rank, target=None)
        
                Put data into a memory window on a remote process.
        """
        pass

    def Set_attr(self, keyval, attrval): # real signature unknown; restored from __doc__
        """
        Win.Set_attr(self, int keyval, attrval)
        
                Store attribute value associated with a key
        """
        pass

    def Set_errhandler(self, Errhandler_errhandler): # real signature unknown; restored from __doc__
        """
        Win.Set_errhandler(self, Errhandler errhandler)
        
                Set the error handler for a window
        """
        pass

    def Set_info(self, Info_info): # real signature unknown; restored from __doc__
        """
        Win.Set_info(self, Info info)
        
                Set new values for the hints
                associated with a window
        """
        pass

    def Set_name(self, name): # real signature unknown; restored from __doc__
        """
        Win.Set_name(self, name)
        
                Set the print name associated with the window
        """
        pass

    def Shared_query(self, rank): # real signature unknown; restored from __doc__
        """
        Win.Shared_query(self, int rank)
        
                Query the process-local address
                for  remote memory segments
                created with `Win.Allocate_shared()`
        """
        pass

    def Start(self, group, assertion=0): # real signature unknown; restored from __doc__
        """
        Win.Start(self, Group group, int assertion=0)
        
                Start an RMA access epoch for MPI
        """
        pass

    def Sync(self): # real signature unknown; restored from __doc__
        """
        Win.Sync(self)
        
                Synchronize public and private copies of the given window
        """
        pass

    def Test(self): # real signature unknown; restored from __doc__
        """
        Win.Test(self)
        
                Test whether an RMA exposure epoch has completed
        """
        pass

    def Unlock(self, rank): # real signature unknown; restored from __doc__
        """
        Win.Unlock(self, int rank)
        
                Complete an RMA access epoch at the target process
        """
        pass

    def Unlock_all(self): # real signature unknown; restored from __doc__
        """
        Win.Unlock_all(self)
        
                Complete an RMA access epoch at all processes
        """
        pass

    def Wait(self): # real signature unknown; restored from __doc__
        """
        Win.Wait(self)
        
                Complete an RMA exposure epoch begun with `Win.Post()`
        """
        pass

    def __bool__(self, *args, **kwargs): # real signature unknown
        """ self != 0 """
        pass

    def __eq__(self, *args, **kwargs): # real signature unknown
        """ Return self==value. """
        pass

    def __ge__(self, *args, **kwargs): # real signature unknown
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs): # real signature unknown
        """ Return self>value. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    def __le__(self, *args, **kwargs): # real signature unknown
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs): # real signature unknown
        """ Return self<value. """
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs): # real signature unknown
        """ Return self!=value. """
        pass

    attrs = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window attributes"""

    flavor = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window create flavor"""

    group = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window group"""

    info = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window info"""

    memory = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window memory buffer"""

    model = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window memory model"""

    name = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """window name"""


    __hash__ = None


# variables with complex values

AINT = None # (!) real value is ''

BOOL = None # (!) forward: C_BOOL, real value is ''

BYTE = None # (!) real value is ''

CHAR = None # (!) real value is ''

CHARACTER = None # (!) real value is ''

COMM_NULL = None # (!) real value is ''

COMM_SELF = None # (!) real value is ''

COMM_WORLD = None # (!) real value is ''

COMPLEX = None # (!) forward: F_COMPLEX, real value is ''

COMPLEX16 = None # (!) real value is ''

COMPLEX32 = None # (!) real value is ''

COMPLEX4 = None # (!) real value is ''

COMPLEX8 = None # (!) real value is ''

COUNT = None # (!) real value is ''

CXX_BOOL = None # (!) real value is ''

CXX_DOUBLE_COMPLEX = None # (!) real value is ''

CXX_FLOAT_COMPLEX = None # (!) real value is ''

CXX_LONG_DOUBLE_COMPLEX = None # (!) real value is ''

C_BOOL = None # (!) real value is ''

C_COMPLEX = None # (!) real value is ''

C_DOUBLE_COMPLEX = None # (!) real value is ''

C_FLOAT_COMPLEX = None # (!) real value is ''

C_LONG_DOUBLE_COMPLEX = None # (!) real value is ''

DATATYPE_NULL = None # (!) real value is ''

DOUBLE = None # (!) real value is ''

DOUBLE_COMPLEX = None # (!) real value is ''

DOUBLE_INT = None # (!) real value is ''

DOUBLE_PRECISION = None # (!) forward: F_DOUBLE, real value is ''

ERRHANDLER_NULL = None # (!) real value is ''

ERRORS_ARE_FATAL = None # (!) real value is ''

ERRORS_RETURN = None # (!) real value is ''

FILE_NULL = None # (!) real value is ''

FLOAT = None # (!) real value is ''

FLOAT_INT = None # (!) real value is ''

F_BOOL = None # (!) real value is ''

F_COMPLEX = None # (!) real value is ''

F_DOUBLE = None # (!) real value is ''

F_DOUBLE_COMPLEX = DOUBLE_COMPLEX

F_FLOAT = None # (!) real value is ''

F_FLOAT_COMPLEX = F_COMPLEX

F_INT = None # (!) real value is ''

GROUP_EMPTY = None # (!) real value is ''

GROUP_NULL = None # (!) real value is ''

INFO_ENV = None # (!) real value is ''

INFO_NULL = None # (!) real value is ''

INT = None # (!) forward: SIGNED_INT, real value is ''

INT16_T = None # (!) real value is ''

INT32_T = None # (!) real value is ''

INT64_T = None # (!) real value is ''

INT8_T = None # (!) forward: SINT8_T, real value is ''

INTEGER = F_INT

INTEGER1 = None # (!) real value is ''

INTEGER16 = None # (!) real value is ''

INTEGER2 = None # (!) real value is ''

INTEGER4 = None # (!) real value is ''

INTEGER8 = None # (!) real value is ''

INT_INT = None # (!) forward: TWOINT, real value is ''

LB = None # (!) real value is ''

LOGICAL = F_BOOL

LOGICAL1 = None # (!) real value is ''

LOGICAL2 = None # (!) real value is ''

LOGICAL4 = None # (!) real value is ''

LOGICAL8 = None # (!) real value is ''

LONG = None # (!) forward: SIGNED_LONG, real value is ''

LONG_DOUBLE = None # (!) real value is ''

LONG_DOUBLE_INT = None # (!) real value is ''

LONG_INT = None # (!) real value is ''

LONG_LONG = None # (!) forward: SIGNED_LONG_LONG, real value is ''

MESSAGE_NO_PROC = None # (!) real value is ''

MESSAGE_NULL = None # (!) real value is ''

OFFSET = None # (!) real value is ''

PACKED = None # (!) real value is ''

pickle = None # (!) real value is ''

REAL = F_FLOAT

REAL16 = None # (!) real value is ''

REAL2 = None # (!) real value is ''

REAL4 = None # (!) real value is ''

REAL8 = None # (!) real value is ''

REQUEST_NULL = None # (!) real value is ''

SHORT = None # (!) forward: SIGNED_SHORT, real value is ''

SHORT_INT = None # (!) real value is ''

SIGNED_CHAR = None # (!) real value is ''

SIGNED_INT = None # (!) real value is ''

SIGNED_LONG = None # (!) real value is ''

SIGNED_LONG_LONG = None # (!) real value is ''

SIGNED_SHORT = None # (!) real value is ''

SINT16_T = INT16_T

SINT32_T = INT32_T

SINT64_T = INT64_T

SINT8_T = None # (!) real value is ''

TWOINT = None # (!) real value is ''

UB = None # (!) real value is ''

UINT16_T = None # (!) real value is ''

UINT32_T = None # (!) real value is ''

UINT64_T = None # (!) real value is ''

UINT8_T = None # (!) real value is ''

UNSIGNED = None # (!) real value is ''

UNSIGNED_CHAR = None # (!) real value is ''

UNSIGNED_INT = UNSIGNED

UNSIGNED_LONG = None # (!) real value is ''

UNSIGNED_LONG_LONG = None # (!) real value is ''

UNSIGNED_SHORT = None # (!) real value is ''

WCHAR = None # (!) real value is ''

WIN_NULL = None # (!) real value is ''

_typedict = {
    '?': C_BOOL,
    'B': UNSIGNED_CHAR,
    'D': C_DOUBLE_COMPLEX,
    'F': C_FLOAT_COMPLEX,
    'G': C_LONG_DOUBLE_COMPLEX,
    'H': UNSIGNED_SHORT,
    'I': UNSIGNED,
    'L': UNSIGNED_LONG,
    'Q': UNSIGNED_LONG_LONG,
    'S': CHAR,
    'Zd': '<value is a self-reference, replaced by this string>',
    'Zf': '<value is a self-reference, replaced by this string>',
    'Zg': '<value is a self-reference, replaced by this string>',
    'b': SIGNED_CHAR,
    'c': '<value is a self-reference, replaced by this string>',
    'd': DOUBLE,
    'f': FLOAT,
    'g': LONG_DOUBLE,
    'h': SIGNED_SHORT,
    'i': SIGNED_INT,
    'l': SIGNED_LONG,
    'p': AINT,
    'q': SIGNED_LONG_LONG,
}

_typedict_c = {
    '?': C_BOOL,
    'B': UNSIGNED_CHAR,
    'D': C_DOUBLE_COMPLEX,
    'F': C_FLOAT_COMPLEX,
    'G': C_LONG_DOUBLE_COMPLEX,
    'H': UNSIGNED_SHORT,
    'I': UNSIGNED,
    'L': UNSIGNED_LONG,
    'Q': UNSIGNED_LONG_LONG,
    'b': SIGNED_CHAR,
    'c16': '<value is a self-reference, replaced by this string>',
    'c32': '<value is a self-reference, replaced by this string>',
    'c8': '<value is a self-reference, replaced by this string>',
    'd': DOUBLE,
    'f': FLOAT,
    'f16': LONG_DOUBLE,
    'f4': '<value is a self-reference, replaced by this string>',
    'f8': '<value is a self-reference, replaced by this string>',
    'g': '<value is a self-reference, replaced by this string>',
    'h': SIGNED_SHORT,
    'i': SIGNED_INT,
    'i1': SINT8_T,
    'i2': INT16_T,
    'i4': INT32_T,
    'i8': INT64_T,
    'l': SIGNED_LONG,
    'q': SIGNED_LONG_LONG,
    'u1': UINT8_T,
    'u2': UINT16_T,
    'u4': UINT32_T,
    'u8': UINT64_T,
}

_typedict_f = {
    '?': F_BOOL,
    '?1': LOGICAL1,
    '?2': LOGICAL2,
    '?4': LOGICAL4,
    '?8': LOGICAL8,
    'c': F_COMPLEX,
    'c16': COMPLEX16,
    'c32': COMPLEX32,
    'c8': COMPLEX8,
    'd': F_DOUBLE,
    'i': F_INT,
    'i1': INTEGER1,
    'i2': INTEGER2,
    'i4': INTEGER4,
    'i8': INTEGER8,
    'r': F_FLOAT,
    'r16': REAL16,
    'r4': REAL4,
    'r8': REAL8,
    's': '<value is a self-reference, replaced by this string>',
    'z': DOUBLE_COMPLEX,
}

__loader__ = None # (!) real value is ''

__pyx_capi__ = {
    'PyMPIComm_Get': None, # (!) real value is ''
    'PyMPIComm_New': None, # (!) real value is ''
    'PyMPIDatatype_Get': None, # (!) real value is ''
    'PyMPIDatatype_New': None, # (!) real value is ''
    'PyMPIErrhandler_Get': None, # (!) real value is ''
    'PyMPIErrhandler_New': None, # (!) real value is ''
    'PyMPIFile_Get': None, # (!) real value is ''
    'PyMPIFile_New': None, # (!) real value is ''
    'PyMPIGroup_Get': None, # (!) real value is ''
    'PyMPIGroup_New': None, # (!) real value is ''
    'PyMPIInfo_Get': None, # (!) real value is ''
    'PyMPIInfo_New': None, # (!) real value is ''
    'PyMPIMessage_Get': None, # (!) real value is ''
    'PyMPIMessage_New': None, # (!) real value is ''
    'PyMPIOp_Get': None, # (!) real value is ''
    'PyMPIOp_New': None, # (!) real value is ''
    'PyMPIRequest_Get': None, # (!) real value is ''
    'PyMPIRequest_New': None, # (!) real value is ''
    'PyMPIStatus_Get': None, # (!) real value is ''
    'PyMPIStatus_New': None, # (!) real value is ''
    'PyMPIWin_Get': None, # (!) real value is ''
    'PyMPIWin_New': None, # (!) real value is ''
}

__spec__ = None # (!) real value is ''

