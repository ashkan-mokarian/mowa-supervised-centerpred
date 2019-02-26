import os
import random
import logging
from time import sleep

logging.getLogger(__name__)


class GeneratorForTfData:
    """Base class for data generator to pass to tf...from_generator

    Implicitely assumes several porperties, such as self.list :(
    Subclass must implement create_list(), provide_data(el), and call super(
    ).__init__(is_training) at the end of its __init__()
    """

    def __init__(self, is_training):
        self.is_training = is_training

    def __iter__(self):
        self.index = -1
        self.list = self.create_list()
        return self

    def __call__(self, *args, **kwargs):
        # Because tf.data1.Dataset.from_generator expects a callable object
        return self.__iter__()

    def __next__(self):
        self.index += 1
        # Shuffle before starting the list, implicitly does not shuffle for
        # the first epoch, maybe helpful for debugging
        if self.index == len(self.list):
            if not self.is_training:
                raise StopIteration
            random.shuffle(self.list)
            self.index = 0
        return self.provide_data(self.list[self.index])

    def provide_data(self, el=None):
        raise NotImplementedError

    def create_list(self):
        raise NotImplementedError


class GeneratorFromFileList(GeneratorForTfData):

    def __init__(self,
                 data_dir_or_file_list,
                 is_training):
        self.file_list = data_dir_or_file_list
        super().__init__(is_training)

    def create_list(self):
        data_dir_or_file_list = self.file_list
        if isinstance(data_dir_or_file_list, str):
            assert os.path.isdir(data_dir_or_file_list), \
                'dir:{} is not a valid path'.format(data_dir_or_file_list)
            return [os.path.join(data_dir_or_file_list, s) for s in os.listdir(data_dir_or_file_list)]
        else:
            assert all([os.path.isfile(s) for s in data_dir_or_file_list]), \
                'Some of the files do not exist'
            return data_dir_or_file_list

    def provide_data(self, file):
        raise NotImplementedError


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    class myGen(GeneratorFromFileList):
        def provide_data(self, file):
            return file
    datagen = myGen('data/test', True)
    import tensorflow as tf
    ds = tf.data.Dataset.from_generator(datagen, tf.string)
    it = ds.make_one_shot_iterator()
    el = it.get_next()
    s = tf.Session()
    while True:
        try:
            print(s.run(el))
            sleep(1)
        except:
            break
