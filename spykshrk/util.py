import numpy as np
import enum
import collections


class AttrDict(collections.UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if hasattr(self, 'data') and (key is not 'data'):

            self.data[key] = value

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            try:
                return self[item]
            except KeyError:
                raise AttributeError(item)

    def __getitem__(self, key):
        if hasattr(self, 'data'):
            return super().__getitem__(key)

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"


class AttrDictEnum(AttrDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as kerr:
            retval = []
            keylist = []
            for key, val in self.__dict__['data'].items():
                try:
                    if item == key.name:
                        keylist = [key]
                        retval.append(val)
                except AttributeError:
                    if item == key:
                        keylist = [key]
                        retval.append(val)
            if len(retval) > 1:
                raise KeyError('Found multiple keys, possible overlapping enums: ' + keylist)
            elif len(retval) == 0:
                raise KeyError(item)
            return retval[0]


class Groupby:
    def __init__(self, data, keys):
        self.data = data
        self.keys = keys
        _, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result

    def __iter__(self):
        for inds in self.indices:
            yield (self.keys[inds[0]], self.data[inds])


class EnumMapping(enum.Enum):
    def __eq__(self, other):
        if isinstance(other, str):
            try:
                return self == self.__getattr(other)
            except AttributeError:
                return False
        else:
            return self.value == other



