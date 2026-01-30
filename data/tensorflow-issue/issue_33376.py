import gc


class TFImporter:
    def __init__(self, name):
        self._name = name
        print(f"TFImporter init {self._name}")

    def get_tf(self):
        print(f"import tensorflow {self._name}")
        import tensorflow
        print(tensorflow.version.VERSION)

    def get_other_module(self):
        print(f"import logging {self._name}")
        import logging
        logging.info("Message")

    def __del__(self):
        print(f"TFImporter delete {self._name}")


def main():
    importer1 = TFImporter(1)
    importer1.get_other_module()
    del importer1
    print("importer1 deleted")

    importer2 = TFImporter(2)
    importer2.get_tf()
    del importer2
    print("importer2 deleted")

    importer3 = TFImporter(3)
    importer3.get_tf()
    del importer3
    print("importer3 deleted")

    print(f"Garbage collection: {gc.collect()}")

    print(f"Waiting for input:")
    input()


main()