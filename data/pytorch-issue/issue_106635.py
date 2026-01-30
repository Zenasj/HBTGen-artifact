from pathlib import Path
import tensorflow as tf


class GFileWrapper(tf.io.gfile.GFile):
    def __init__(self, path, mode="r") -> None:
        super().__init__(path, mode)
        
    def write(self, data):
        return super().write(bytes(data))
    
    # a not quite efficient readinto, but it works
    def readinto(self, buffer):
        # read up to buffer's length
        data = self.read(len(buffer))
        length = len(data)
        buffer[:length] = data
        return length


class HdfsPath(type(Path())):
    def __new__(cls, *pathsegments):
        return super().__new__(cls, *pathsegments)
    
    @staticmethod
    def _fix_path(path):
        path = str(path)
        if path.startswith("hdfs:/") and not path.startswith("hdfs://"):
          path = path.replace("hdfs:/", "hdfs://")
        return path

    def open(self, mode="r", *args, **kwargs):
        return GFileWrapper(HdfsPath._fix_path(self), mode=mode)
    
    def mkdir(self, **kwargs) -> None:
        return tf.io.gfile.makedirs(HdfsPath._fix_path(self))
    
    def rename(self, target):
        return tf.io.gfile.rename(HdfsPath._fix_path(self), HdfsPath._fix_path(target))

writer = FileSystemWriter(HdfsPath("hdfs://..."), sync_files=False)
reader = FileSystemReader(HdfsPath("hdfs://..."))