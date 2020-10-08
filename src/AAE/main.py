from ResNet import ResNet
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='tiny', help='[cifar10, cifar100, mnist, fashion-mnist, tiny')


    parser.add_argument('--epoch', type=int, default=82, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch per gpu')
    parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--bootstrap_dir', type=str, default=None,
                        help='Directory name to load the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    parser.add_argument('--ae', action='store_true', help='Train autoencoder instead of classifier')
    parser.add_argument('--aae', action='store_true', help='Tran augmented autoencoder')
    parser.add_argument('--hourglass', action='store_true', help='Hourglass Architecture')
    parser.add_argument('--iteration', type=int, default=-1, help='Total number of iterations per epoch')
    parser.add_argument('--out', default=None, help='Output file/directory')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    if args.aae:
        args.ae = True

    # open session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        cnn = ResNet(sess, args)

        # build graph
        cnn.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            # launch the graph in a session
            cnn.train()

            print(" [*] Training finished! \n")

            cnn.test()
            print(" [*] Test finished!")
        elif args.phase == 'test' :
            cnn.test()
            print(" [*] Test finished!")
        elif args.phase == 'peek':
            cnn.peek()
        else:
            assert False, "unknonw phase {}".format(args.phase)

if __name__ == '__main__':
    main()
