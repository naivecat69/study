import numpy as np
import os, sys, argparse, warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', 'invalid value encountered in add')

class decodeCC:
    def __init__(self, args):
        self.ifile = args.ifile
        self.ofile = args.ofile
        self.K = args.K
        self.num_states = 2 ** (args.K - 1)
        self.generator = np.array([args.g1, args.g2])
        self.mode = args.mode
        if args.punc:
            self.punc = list(map(int, args.punc.split(',')))


# ===== Main Function ==================================================
if __name__ == '__main__':
    # argparse를 이용해서 인자값을 선언받습니다.
    parser = argparse.ArgumentParser(description="""
    \n\n\n\n\n
    #######################################
    #   DecodeCC with Viberbi Algorithm   #
    #######################################
    \n\n\n\n\n
    Usage: -i <filename> -o <outputfilename> -k <constraint K> -g1 <poly1 (type: oct)> -g2 <poly2 (type: oct)> -mode <Decode mode>
    
    *** Decode mode: hard, soft
    *** Decode type: truncated, terminated
    
    for example, filename: test.bin // K = 4 //poly1: x^3+1 // poly2: x^2+1 // hard // truncated
    --------------------------------------------------------------
    $python decodeCC.py -i test.bin -o decoded.bin -k 4 -g1 11 -g2 5 -mode hard -type truncated ((-punc 0,1,0,1))
    --------------------------------------------------------------
    
    """)
    parser.add_argument("-i", "--ifile", type = str, required=True, help="Input file")
    parser.add_argument("-o", "--ofile", type = str, required=True, help="Output file")
    parser.add_argument("-k", "--k", type = int, required=True, help="constraint length K")
    parser.add_argument("-g1", "--poly1", type = str, required=True, help="Generate Polynomial1")
    parser.add_argument("-g2", "--poly2", type = str, required=True, help="Generate Polynomial2")
    parser.add_argument("-mode", "--mode", type = str, required=True, default="truncated", help="Decode mode")
    parser.add_argument('-punc', '--puncturing', type = str, required=False, default=False, help="Puncturing")

    args = parser.parse_args()

    decodeCC = decodeCC(args)