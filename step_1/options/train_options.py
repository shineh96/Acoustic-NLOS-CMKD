from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self) 

        #self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of displaying average loss')
        self.parser.add_argument('--epochs', type=int, default=500, help='train epochs')
        #self.parser.add_argument('--learning_rate_decrease_iter', type=int, default=-1, help='how often is the learning rate decreased')
        #self.parser.add_argument('--decay_factor', type=float, default=0.95, help='learning rate decay factor')
        #self.parser.add_argument('--validation_freq', type=int, default=50, help='frequency of testing on validation set')
        #self.parser.add_argument('--epoch_save_freq', type=int, default=5, help='frequency of saving intermediate models')
        # optimizer arguments  
        self.parser.add_argument('--learning_rate', type=int, default=0.0001, help='learning rate')
        #self.parser.add_argument('--weight_decay', type=float, default=0.0005, help='weights regularizer')
        
        self.mode = 'train'
        self.isTrain = True
        