from argparse import ArgumentParser
import measurementsub
def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-d", "--lead", required=False, type=str,
                        default="ii",
                        help="lead name")
    parser.add_argument("-s", "--max", required=False, type=int,
                        default=32,
                        help="extended time duration in sec")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input directory")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output directory")

    parser.add_argument("-ih", "--input_html", type=str, required=True,
                        help="input html report")

    return parser

def main():
    args = build_argparser().parse_args()
    measurementsub.measurementMain(args.input, args.lead, args.max, args.output, args.input_html)


if __name__ == '__main__':
    main()
    exit(0)