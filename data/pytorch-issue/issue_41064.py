import random

class ChunkDataset(Dataset):
    def __init__(self, feat_scp_file, chunk_size_range=(100, 500)):
        super(ChunkDataset, self).__init__()
        self.feat_scp_file = feat_scp_file

        self.feature_reader = SynchronizedFeatureReader(self.feat_scp_file)
        self.utt_list = self.feature_reader.get_utt_list()
        self.min_chunk_size = chunk_size_range[0]
        self.max_chunk_size = chunk_size_range[1]

    def __len__(self):
        return len(self.feature_reader)

    def __getitem__(self, item):
        utt_id = self.utt_list[item]
        feat = self.feature_reader[utt_id]
        feat_len = feat.shape[0]

        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        chunk_start = random.randint(0, max(0, feat_len - chunk_size))

        return feat[chunk_start: min(chunk_start + chunk_size, feat_len), :]

class SynchronizedFeatureReader(object):
    def __init__(self, scp_file):
        self.scp_file = scp_file
        self.feat_dict = ScriptReader(scp_file)

    def _load(self, utt_id):
        return self.feat_dict[utt_id]

    def __len__(self):
        return len(self.feat_dict)

    def __getitem__(self, item):
        return self.feat_dict[item]

    def __iter__(self):
        for (utt_id, feat) in self.feat_dict:
            yield utt_id, feat

    def get_utt_list(self):
        return self.feat_dict.index_keys

class ScriptReader(Reader):
    def __init__(self, ark_scp):
        self.fmgr = dict()
        def addr_processor(addr):
            addr_token = addr.split(":")
            if len(addr_token) == 1:
                raise ValueError("Unsupported scripts address format")
            path, offset = ":".join(addr_token[0:-1]), int(addr_token[-1])
            return (path, offset)

        super(ScriptReader, self).__init__(ark_scp,
                                           value_processor=addr_processor)

    def __del__(self):
        for name in self.fmgr:
            self.fmgr[name].close()

    def _open(self, obj, addr):
        if obj not in self.fmgr:
            self.fmgr[obj] = open(obj, "rb")
        arkf = self.fmgr[obj]
        arkf.seek(addr)
        return arkf

    def _load(self, key):
        path, addr = self.index_dict[key]
        fd = self._open(path, addr) 
        obj = io.read_float_mat_vec(fd, direct_access=True) 
        return obj

def expect_binary(fd):
    flags = bytes.decode(fd.read(2))
    throw_on_error(flags == '\0B', f'Expect binary flag, but gets {flags}')

class ScriptReader(Reader):
    def __init__(self, ark_scp):
        self.fmgr = dict()
        self.main_lock = Lock()

        def addr_processor(addr):
            addr_token = addr.split(":")
            if len(addr_token) == 1:
                raise ValueError("Unsupported scripts address format")
            path, offset = ":".join(addr_token[0:-1]), int(addr_token[-1])
            return (path, offset)

        super(ScriptReader, self).__init__(ark_scp,
                                                              value_processor=addr_processor)

    def __del__(self):
        for name in self.fmgr:
            self.fmgr[name]['handle'].close()

    def _load(self, key):
        path, addr = self.index_dict[key]

        self.main_lock.acquire()
        if path not in self.fmgr:
            # print("Create new lock")
            self.fmgr[path] = dict()
            self.fmgr[path]['handle'] = open(path, "rb")
            self.fmgr[path]['lock'] = Lock()
        self.main_lock.release()

        lock = self.fmgr[path]['lock']
        lock.acquire()
        fd = self.fmgr[path]['handle']
        fd.seek(addr)
        obj = io.read_float_mat_vec(fd, direct_access=True)
        lock.release()
        return obj

def load_utt(reader, utt_id):
        return reader[utt_id].shape

def test_multiprocess_script_reader(scp, max_num = 100):
    # test ScriptReader
    scp_reader = ScriptReader(scp)
    pool = Pool(processes=8)
    try:
        utt_list = scp_reader.index_keys
        result_list = list()
        num_done = 0
        for utt_id in utt_list:
            result = pool.apply_async(load_utt, (scp_reader, utt_id))
            result_list.append(result)
            num_done += 1
            if num_done > max_num:
                break
        pool.close()
        pool.join()
        print("script_reader finished!")
    except TypeError as e:
        print("Using ScriptReader leads to the error:\n", e)
    finally:
        del scp_reader
        del pool

    # test SynchronizedScriptReader
    scp_reader = SynchronizedScriptReader(scp)
    pool = Pool(processes=8)
    try:
        utt_list = scp_reader.index_keys
        result_list = list()
        num_done = 0
        for utt_id in utt_list:
            result = pool.apply_async(load_utt, (scp_reader, utt_id))
            result_list.append(result)
            num_done += 1
            if num_done > max_num:
                break
        pool.close()
        pool.join()
        print("synchronized_script_reader finished!")
    except TypeError as e:
        print("Using SynchronizedScriptReader leads to the error:\n", e)
    finally:
        del scp_reader
        del pool

    print("TEST *multiprocess_script_reader* DONE!")

for index, (data, label) in enumerate(data_loader):
     # somethings
     print(f"data = {data.shape}")
     pass