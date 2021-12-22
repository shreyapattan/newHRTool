import numpy as np
from os import walk
import wfdb
import os
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import pandas as pd
import subprocess

def print_summary(rms_min,rms_max,rms_mean):
    print('rms error min  HR: %d' % rms_min)
    print('rms error mean HR: %d' % rms_mean)
    print('rms error max  HR: %d\n' % rms_max)

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def getvalus(parser,idname):
    tag = parser.find(id=idname)
    valusOut = tag.text.split(" ")[0]
    try:
        res = float(valusOut)
    except Exception as e:
        res = 0
    return res
def createReport(dataout,maxdata,meandata):
    writer = pd.ExcelWriter('HR_Report.xlsx', engine='xlsxwriter')
    df = pd.DataFrame(dataout)
    df_max = pd.DataFrame(maxdata)
    df_mean = pd.DataFrame(meandata)

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='HR MIN')
    df_max.to_excel(writer, sheet_name='HR Max')
    df_mean.to_excel(writer, sheet_name='HR Mean')

    writer.save()

def hrComparison(only_name,full_path ,htmpath):
    hr_ext = "htr"
    ref_ext = "atr"

    htmlFulPath = os.path.join(htmpath, only_name, only_name + ".html")
    htr_path = full_path+"."+hr_ext
    mean_data =[]
    min_data =[]
    max_data =[]
    flag = False

    if os.path.isfile(htr_path) and os.path.isfile(htmlFulPath):
        try:
            annotationFull = wfdb.rdann(full_path, hr_ext)
            annotation = annotationFull.__dict__
            symbolData = annotation['symbol']
            aux_ = annotation['aux_note']
            annotation_atr = wfdb.rdann(full_path, ref_ext)
            annot_atr = annotation_atr.__dict__
            symbolData_atr = annot_atr['symbol']

            ref_hr = []
            flag = False
            listsym = ["L", "R", "N"]
            for count_t, val_t in enumerate(symbolData_atr):
                if symbolData[count_t] == "=":
                    hrlist_flag = True
                    if flag:
                        flag = False
                        hrlist_flag = False
                    if val_t in listsym:
                        pass
                    else:
                        flag = True
                        hrlist_flag = False

                    if hrlist_flag:
                        ref_hr.append(float(aux_[count_t]))
            if os.path.isfile(htmlFulPath):
                HTMLFile = open(htmlFulPath, "r")
                index = HTMLFile.read()
                ParseCur = BeautifulSoup(index, 'lxml')
                averhrRE = getvalus(ParseCur, "average-hr")
                maxhr = getvalus(ParseCur, "maximum-hr")
                minhr = getvalus(ParseCur, "minimum-hr")
                ref_hr_ = (np.array(ref_hr))
                ref_min = round(np.min(ref_hr_))
                ref_max = round(np.max(ref_hr))
                ref_mean = round(np.mean(ref_hr))
                calculated_array = [minhr,averhrRE, maxhr ]

                rms_min = rmse(np.array(ref_min), np.array(calculated_array[0]))
                rms_max = rmse(np.array(ref_max), np.array(calculated_array[2]))
                rms_mean = rmse(np.array(ref_mean), np.array(calculated_array[1]))
                print(only_name +" Reference : ", round(ref_min),round(ref_mean), round(ref_max) )
                print(only_name +" AccurECG  : ", minhr,averhrRE, maxhr)
                print_summary(rms_min, rms_max, rms_mean)

                min_data =[only_name,int(round(ref_min)),int(minhr),rms_min]
                max_data =[only_name,int(round(ref_max)),int(maxhr),rms_max]
                mean_data = [only_name, int(round(ref_mean)), int(averhrRE), rms_mean]
                flag = True
        except Exception as e:
            print("error :",only_name)
    return min_data,max_data,mean_data,flag


def creat_htr(fileDir,cygwin_bat,only_name):
    HR_tolerance = "500"
    db_name = fileDir#os.path.basename(fileDir)
    ref_dir = os.path.join(fileDir, 'ref')
    if os.path.isdir(ref_dir):
        pass
    else:
        os.makedirs(ref_dir)

    p = subprocess.Popen(cygwin_bat, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, universal_newlines=True)
    cmd_1 = 'cd ' + '"'+ db_name +'"'+ ' && rdann -r ' + only_name + " -a atr>ref/" + only_name + '.txt'
    print(cmd_1)
    out = p.communicate(input=cmd_1)
    p.terminate()

    p1 = subprocess.Popen(cygwin_bat, shell=True, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, universal_newlines=True)
    cmd_2 = 'cd ' + '"'+ db_name +'"'+ ' && ihr -r ' + only_name + " -a atr -d " + HR_tolerance + " -V -i >ref/" + only_name + '_hr.txt'

    print(cmd_2)

    out1 = p1.communicate(input=cmd_2)
    p1.terminate()
    atr_file = os.path.join(ref_dir, only_name + ".txt")
    f = open(atr_file, "r")
    atr_data = f.read()
    f.close()
    hr_file = os.path.join(ref_dir, only_name + "_hr.txt")
    f_hr = open(hr_file, "r")
    hr_data = f_hr.read()
    f_hr.close()
    atr_list = atr_data.split('\n')
    atr_final = []
    atr_index = []
    for x in atr_list:
        if len(x) > 0:
            out = x.split('\t')
            res = out[0].split()
            if len(out) > 1:
                final = res + [out[1]]
            else:
                final = res + ['']
            atr_index.append(int(final[1]))
            atr_final.append(final)
    atr_index = np.array(atr_index)
    hr_final = []

    hr_list = hr_data.split('\n')
    for x_h in hr_list:
        if len(x_h) > 0:
            out = x_h.split('\t')
            index = int(out[0])
            result = np.where(atr_index == index)
            if len(result[0]):
                replace_ind = result[0][0]
                atr_final[replace_ind][2] = "="
                atr_final[replace_ind][6] = out[1]
            hr_final.append(out)
    atr_final = np.array(atr_final)

    location = atr_final[:, 1].astype(np.int)
    symbol = atr_final[:, 2]
    aux = atr_final[:, 6]
    ann1 = wfdb.Annotation(record_name=only_name, extension='htr',
                           sample=location, symbol=symbol, aux_note=aux.tolist())
    ann1.wrann(write_fs=True, write_dir=fileDir)

    return "done"
def mainHR(inputpath,cygwin_bat,htmpath):
    finalData_min = {"Name": {}, "Ref": {}, "test": {}, "Rms error": {}}
    maxdata_f = {"Name": {}, "Ref":{}, "test": {}, "Rms error": {}}
    meandata_f = {"Name": {}, "Ref": {}, "test": {}, "Rms error": {}}
    count = 0
    for (dirpath, dirnames, filenames) in walk(inputpath):
        for xname in filenames:
            npath = os.path.join(dirpath, xname)
            fullPath, file_extension = os.path.splitext(npath)
            only_name = os.path.basename(fullPath)
            if file_extension ==".dat":
                if os.path.isfile(fullPath + ".dat") and os.path.isfile(fullPath + ".hea" ) and os.path.isfile(fullPath + ".atr" ):
    
                    creat_htr(inputpath, cygwin_bat,only_name)
    
                    min_data,max_data,mean_data,flag = hrComparison(only_name, fullPath, htmpath)
                    if flag:

                        finalData_min["Name"][count] = min_data[0]
                        finalData_min["Ref"][count] = min_data[1]
                        finalData_min["test"][count] = min_data[2]
                        finalData_min["Rms error"][count] = min_data[3]

                        maxdata_f["Name"][count] =max_data[0]
                        maxdata_f["Ref"][count] =max_data[1]
                        maxdata_f["test"][count] =max_data[2]
                        maxdata_f["Rms error"][count] =max_data[3]

                        meandata_f["Name"][count] =mean_data[0]
                        meandata_f["Ref"][count] =mean_data[1]
                        meandata_f["test"][count] =mean_data[2]
                        meandata_f["Rms error"][count] =mean_data[3]
                        count +=1

    createReport(finalData_min,maxdata_f,meandata_f)

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input reference directory")
    parser.add_argument("-ih", "--input_html", type=str, required=True,
                        help="input html report")
    parser.add_argument("-c", "--cygwin_bat", type=str, required=True,
                        help="cygwin bat location")

    return parser
#min avg max erf
def main():
    args = build_argparser().parse_args()

    #mainHR(inputpath,cygwin_bat,htmpath)

    mainHR(args.input,  args.cygwin_bat,args.input_html)


if __name__ == '__main__':
    main()
    exit(0)


