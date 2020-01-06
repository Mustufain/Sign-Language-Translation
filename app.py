import sys
from utils.sign_utils import DataPreperation
from experiments.inception import CustomInceptionV3
from asl_translation.hand_translation import HandTranslator
import os

if __name__ == "__main__":
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        if argument == 'prepare_data':
            data = 'asl_digits'
            base_path = os.path.abspath('.') + '/data/'
            data_prepare = DataPreperation(base_path=base_path, data=data)
            data = data_prepare.read_files()
            training, validation, testing = data_prepare.split_files(data)
            print('trainning size', len(training))
            print('validation size', len(validation))
            print('testing size', len(testing))
            train_path = data_prepare.get_train_path()
            valid_path = data_prepare.get_valid_path()
            test_path = data_prepare.get_test_path()

            print('intializing creating datasets...............')
            data_prepare.create_data_set(data=training, path=train_path)
            data_prepare.create_data_set(data=validation, path=valid_path)
            data_prepare.create_data_set(data=testing, path=test_path)

            print('initializing creating h5 datasets.............')

            base_path = data_prepare.get_base_path()

            fname = base_path+"train.h5"
            data_prepare.create_h5_dataset(dir=train_path, fname=fname)
            fname = base_path+"validation.h5"
            data_prepare.create_h5_dataset(dir=valid_path, fname=fname)
            fname = base_path + "test.h5"
            data_prepare.create_h5_dataset(dir=test_path, fname=fname)

            print('creating labels.................')

            fname = base_path+'train_labels.h5'
            data_prepare.create_labels(dir=train_path, fname=fname)
            fname = base_path+'valid_labels.h5'
            data_prepare.create_labels(dir=valid_path, fname=fname)
            fname = base_path + 'test_labels.h5'
            data_prepare.create_labels(dir=test_path, fname=fname)

            print ('saving ids.....................')
            data_prepare.save_id(path=train_path, fname='train_id.json')
            data_prepare.save_id(path=valid_path, fname='valid_id.json')
            data_prepare.save_id(path=test_path, fname='test_id.json')

        if argument == 'train':
            base_dir = os.path.abspath('.') + '/data/'
            inceptionv3 = CustomInceptionV3(base_dir=base_dir, version=1)
            inceptionv3.run()
            inceptionv3.export_model()

    else:
        print('start demo')
        translator = HandTranslator()
        translator.translate()
