import os
import config
import time

from utils import io_data, polygonization


def main():
    filenames = os.listdir(config.DIR_WITH_DATA)
    print("Found ", len(filenames), ' masks for processing.')
    print('Using {} threads.'.format(config.THREADS))
    for filename, i in zip(filenames, range(len(filenames))):
        if not os.path.exists(os.path.join(config.OUTPUT_DIR, filename.split('.')[0] + '.' + config.OUTPUT_EXTENTION)):
            print('Processing mask # ', i, 'started ...')
            start = time.time()
            mask = io_data.read_and_prepare_mask(filename)
            result = polygonization.process_area(mask)
            io_data.save_result(result, filename)
            end = time.time()
            print('Processing mask # ', i, ' done. Elapsed time: ', round(end - start, 2), 'seconds.')
    print('Processing done.')


if __name__ == '__main__':
    main()