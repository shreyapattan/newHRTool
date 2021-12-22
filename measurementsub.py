import shutil
import wfdb
from os import walk
import os
import numpy as np
from datetime import datetime, timedelta
import math
from bs4 import BeautifulSoup
import pandas as pd

def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted
def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))
def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    outD = np.sqrt(mse(actual, predicted))
    return round(outD,3)
def getvalus_(parser,idname):
    tag = parser.find(id=idname)
    valusOut = tag.text.split(" ")[0]
    try:
        out1 = valusOut.split(" sec")
        res = float(out1[0])
    except Exception as e:
        res = 0
    return res
def GetTime(a):
    sec = timedelta(seconds=int(a))
    d = datetime(1,1,1) + sec
    return ("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))


def dataExtend(filename,leadname,overallsec,ResultDir):
    annotationFull = wfdb.rdann(filename, leadname)
    annotationref = annotationFull.__dict__
    sampleData = annotationref['sample']
    annsymbol_Data = annotationref['symbol']
    annsym_subtype = annotationref['subtype']
    annsym_chan = annotationref['chan']
    annsym_num = annotationref['num']
    annsym_aux = annotationref['aux_note']
    name = os.path.basename(filename)
    out_header = wfdb.rdheader(filename)
    x = np.where(np.array(out_header.sig_name) == leadname)

    if x[0]:
        channels_index = x[0][0]
    else:
        channels_index = 0

    sig, fields = wfdb.rdsamp(filename, sampfrom=0, channels=[channels_index])
    sig = sig  # sig.flatten()
    fmtin = out_header.fmt
    in_units = out_header.units
    _comments = out_header.comments
    fs = fields['fs']
    OrgData = np.array(sig)

    sampleDataOrg = np.array(sampleData)
    annsym_subtypeOrg = np.asarray(annsym_subtype)
    annsym_annsym_Dataorg = np.asarray(annsymbol_Data)
    annsym_chanOrg = np.asarray(annsym_chan)
    annsym_numOrg = np.asarray(annsym_num)
    annsym_auxOrg = np.asarray(annsym_aux)

    finaldatacount = overallsec * fs
    coutitration = int(math.ceil(overallsec / (len(OrgData) / fs)))
    finaldata = []

    finaldatasample = []
    finaldatasybtype = []
    symbol_new = []
    chan_new = []
    num_new = []
    aux_new = []

    cind = 0
    for x in range(coutitration):
        breakval = []
        if cind == 0:
            dataout = np.concatenate((OrgData, OrgData), 0)
            adddata = sampleDataOrg + len(OrgData)
            dataoutsample = np.concatenate((sampleDataOrg, adddata), 0)

            dataoutsubtype = np.concatenate((annsym_subtypeOrg, annsym_subtypeOrg), 0)

            dataoutsymb = np.concatenate((annsym_annsym_Dataorg, annsym_annsym_Dataorg), 0)
            dataoutchan = np.concatenate((annsym_chanOrg, annsym_chanOrg), 0)
            dataoutnum = np.concatenate((annsym_numOrg, annsym_numOrg), 0)
            dataoutaux = np.concatenate((annsym_auxOrg, annsym_auxOrg), 0)

        else:
            remainindata = finaldatacount - len(finaldata)
            if remainindata >= len(OrgData):
                dataout = np.concatenate((finaldata, OrgData), 0)
                adddatasap = sampleDataOrg + len(finaldata)
                dataoutsample = np.concatenate((finaldatasample, adddatasap), 0)

                dataoutsubtype = np.concatenate((finaldatasybtype, annsym_subtypeOrg), 0)

                dataoutsymb = np.concatenate((symbol_new, annsym_annsym_Dataorg), 0)
                dataoutchan = np.concatenate((chan_new, annsym_chanOrg), 0)
                dataoutnum = np.concatenate((num_new, annsym_numOrg), 0)
                dataoutaux = np.concatenate((aux_new, annsym_auxOrg), 0)
            else:
                a12 = OrgData[:remainindata]
                dataout = np.concatenate((finaldata, a12), 0)
                newsamAdd = []
                newsubty = []
                newsymb = []
                newcha = []
                newnum = []
                newaux = []

                for counts, value in enumerate(sampleDataOrg):
                    if sampleDataOrg[counts] <= remainindata:
                        newsamAdd.append(sampleDataOrg[counts])
                        newsubty.append(annsym_subtypeOrg[counts])
                        newsymb.append(annsym_annsym_Dataorg[counts])
                        newcha.append(annsym_chanOrg[counts])
                        newnum.append(annsym_numOrg[counts])
                        newaux.append(annsym_auxOrg[counts])

                adddatasap = np.asarray(newsamAdd) + len(finaldata)
                dataoutsample = np.concatenate((finaldatasample, adddatasap), 0)
                dataoutsubtype = np.concatenate((finaldatasybtype, np.asarray(newsubty)), 0)

                dataoutsymb = np.concatenate((symbol_new, np.asarray(newsymb)), 0)
                dataoutchan = np.concatenate((chan_new, np.asarray(newcha)), 0)
                dataoutnum = np.concatenate((num_new, np.asarray(newnum)), 0)
                dataoutaux = np.concatenate((aux_new, np.asarray(newaux)), 0)
                breakval = True

        finaldata = dataout
        finaldatasample = dataoutsample
        finaldatasybtype = dataoutsubtype

        symbol_new = dataoutsymb
        chan_new = dataoutchan
        num_new = dataoutnum
        aux_new = dataoutaux

        cind += 1
        if breakval:
            break

    res = wfdb.wrsamp(name, fs=fs, units=[in_units[0]], sig_name=[leadname], p_signal=finaldata,
                      fmt=[fmtin[0]], write_dir=ResultDir, comments=_comments)
    src = filename+"."+leadname
    dst = os.path.join(ResultDir,name+".ii")
    shutil.copy(src, dst)

    # #print("OutputPath :" + str(os.path.join(ResultDir, name)))
    #
    # finaldatasample1 = [int(item) for item in finaldatasample]
    # finaldatasybtype1 = [int(item) for item in finaldatasybtype]
    # chan_new1 = [int(item) for item in chan_new]
    #
    # ann1 = wfdb.Annotation(record_name=name, extension=leadname,
    #                        sample=np.array(finaldatasample1), symbol=np.array(symbol_new),
    #                        aux_note=aux_new.tolist(),
    #                        subtype=np.asarray(finaldatasybtype1), chan=np.asarray(chan_new1),
    #                        num=num_new)
   # ann1.wrann(write_fs=True, write_dir=ResultDir)

    return ResultDir,filename
def sort_and_index2(arr,ascending): #F_29
    index = [i for i in range(len(arr))]
    bundle = [[o,p] for o,p in zip(arr,index)]
    if ascending:
        bundle.sort()
        sortedArray = [r[0] for r in bundle]
        index = [s[1] for s in bundle]
        return sortedArray,index
    else:
        bundle.sort()
        bundle.reverse()
        sortedArray = [r[0] for r in bundle]
        index = [s[1] for s in bundle]
        return sortedArray,index

def generateRefData(filepath,leadname,ResultDir):

    only_name = os.path.basename(filepath)

    try:
        annotationFull = wfdb.rdann(filepath, leadname)
        annotation = annotationFull.__dict__
        symbolData = annotation['symbol']
        sampleData = annotation['sample']
        sampleAux = annotation['aux_note']
        fs = annotation['fs']

        finalbeat = []
        for count, ele in enumerate(symbolData):
            if ele == "N":
                finalbeat.append(sampleData[count])

        finaldict = {}
        count1 = 0
        for x in symbolData:
            val = ["(", ")"]
            if x not in val:
                if not (len(symbolData) <= count1 + 1):
                    stAE = [sampleData[count1 - 1], sampleData[count1 + 1]]
                    if x not in finaldict:
                        finaldict[x] = [stAE]
                    else:
                        finaldict[x].append(stAE)
            count1 += 1

        if "p" in finaldict and "N" in finaldict:
            P_Data = finaldict["p"]
            N_Data = finaldict["N"]

            pval = []
            pri_val = []
            p_point = []
            pri_point = []
            fpw = {}
            fpri = {}
            for countP, valP in enumerate(P_Data):
                pdis = valP[1] - valP[0]
                pl_ = pdis / fs
                pval.append(pl_)

                p_mid = int(valP[0] + (pdis / 2))
                p_point.append(p_mid)

                fpw[p_mid] = pl_

                for xN in N_Data:
                    nflag = xN[0] > valP[1]
                    if nflag:
                        pri_dis = xN[0] - valP[0]
                        pril_ = pri_dis / fs
                        pri_val.append(pril_)

                        pri_mid = int(valP[0] + (pri_dis / 2))
                        pri_point.append(pri_mid)
                        fpri[pri_mid] = pril_

                        break
            Nval = []
            qrs_point = []
            fqrs = {}
            for countN, valN in enumerate(N_Data):
                Ndis = valN[1] - valN[0]
                vl_ = Ndis / fs
                Nval.append(vl_)
                Q_mid = int(valN[0] + (Ndis / 3))
                fqrs[Q_mid] = vl_
                qrs_point.append(Q_mid)

            final_qrs = np.concatenate((sampleData, np.asarray(qrs_point)), 0)
            final_p = np.concatenate((sampleData, np.asarray(p_point)), 0)
            final_pri = np.concatenate((sampleData, np.asarray(pri_point)), 0)

            sortedfinal, index_q = sort_and_index2(final_qrs, True)
            newsym = []
            aux_q = []
            for counts, idi in enumerate(index_q):
                # print(len(symbolData),idi)
                if len(symbolData) > idi:
                    newsym.append(symbolData[idi])
                    aux_q.append("")
                else:
                    newsym.append("=")
                    auxqi = (fqrs[sortedfinal[counts]])
                    aux_q.append(str(auxqi))

            sortedpwave, index_pw = sort_and_index2(final_p, True)
            new_pwa = []
            aux_pw = []
            for cpw, idipw in enumerate(index_pw):
                # print(len(symbolData),idi)
                if len(symbolData) > idipw:
                    new_pwa.append(symbolData[idipw])
                    aux_pw.append(sampleAux[idipw])
                else:
                    new_pwa.append("=")
                    auxpi = (fpw[sortedpwave[cpw]])
                    aux_pw.append(str(auxpi))

            sortedPri, index_pri = sort_and_index2(final_pri, True)
            newpri = []
            aux_pri = []
            for cpri, idpri in enumerate(index_pri):
                if len(symbolData) > idpri:
                    newpri.append(symbolData[idpri])
                    aux_pri.append(sampleAux[idpri])
                else:
                    newpri.append("=")
                    auxprii = (fpri[sortedPri[cpri]])
                    aux_pri.append(str(auxprii))

            ann1 = wfdb.Annotation(record_name=only_name, extension='q',
                                   sample=np.array(sortedfinal), symbol=(newsym), aux_note=aux_q)
            ann1.wrann(write_fs=True, write_dir=ResultDir)

            sortedpwave1 = [int(item) for item in sortedpwave]

            ann1 = wfdb.Annotation(record_name=only_name, extension='p',
                                   sample=np.array(sortedpwave1), symbol=new_pwa, aux_note=aux_pw)
            ann1.wrann(write_fs=True, write_dir=ResultDir)
            sortedPri1 = [int(item) for item in sortedPri]
            ann1 = wfdb.Annotation(record_name=only_name, extension='pri',
                                   sample=np.array(sortedPri1), symbol=newpri, aux_note=aux_pri)
            ann1.wrann(write_fs=True, write_dir=ResultDir)
        else:
            print("annotation are missing :",only_name)
    except Exception as e:
        print("error :",only_name,e)

    return "Done"
def dataConvert(filename,leadname,overallsec,ResultDir):

    out_header = wfdb.rdheader(filename)
    signal_length = out_header.sig_len
    frequency_per_sec = out_header.fs

    limitdata = overallsec * frequency_per_sec

    if signal_length >= limitdata:
        pass
    else:
        ResultDir_,filename_ = dataExtend(filename, leadname, overallsec, ResultDir)


    return "Done"


def get_ref_data(ext,full_path):
    annotationFull = wfdb.rdann(full_path, ext)
    annotation = annotationFull.__dict__
    aux = annotation['aux_note']
    data = []
    for xp in aux:
        if xp != "":
            data.append(float(xp))
    return data

def compar_tool(htmpath,only_name,full_path,resultpath):
    htmlFulPath = os.path.join(htmpath, only_name, only_name + ".html")
    Dpath = os.path.join(resultpath,only_name)
    p_d = []
    pri_d =[]
    qrs_d =[]
    flag =False
    if os.path.isfile(Dpath+".p") and os.path.isfile(Dpath+".pri") and os.path.isfile(Dpath+".q") and os.path.isfile(htmlFulPath):
        try:
            HTMLFile = open(htmlFulPath, "r")
            index = HTMLFile.read()
            ParseCur = BeautifulSoup(index, 'lxml')
            p_t = getvalus_(ParseCur, "p-interval")
            pri_t = getvalus_(ParseCur, "pr-interval")
            qrs_t = getvalus_(ParseCur, "qrs")

            p_data = get_ref_data("p",Dpath)

            q_data = get_ref_data("q",Dpath)
            pri_data = get_ref_data("pri",Dpath)

            p_mean = np.mean(np.array(p_data))
            rms_p = rmse(np.array(p_mean), np.array(p_t))

            pri_mean = np.mean(np.array(pri_data))
            rms_pri = rmse(np.array(pri_mean), np.array(pri_t))

            q_mean = np.mean(np.array(q_data))
            rms_q = rmse(np.array(q_mean), np.array(qrs_t))

            p_d=[p_mean,p_t,round(rms_p,3)]
            pri_d = [pri_mean, pri_t, round(rms_pri, 3)]
            qrs_d = [q_mean, qrs_t, round(rms_q, 3)]


            flag = True

        except Exception as e:
            print("error", e)

    return flag,p_d,pri_d,qrs_d

def createReport(QrsData,Pdata,PRIdata):
    writer = pd.ExcelWriter('MeasurementReport.xlsx', engine='xlsxwriter')
    df_qrs = pd.DataFrame(QrsData)
    df_P = pd.DataFrame(Pdata)
    df_PRI = pd.DataFrame(PRIdata)

    # Convert the dataframe to an XlsxWriter Excel object.
    df_qrs.to_excel(writer, sheet_name='QRS')
    df_P.to_excel(writer, sheet_name='P Wave')
    df_PRI.to_excel(writer, sheet_name='PRI')

    writer.save()

def measurementMain(inputpath,leadname,overallsec,ResultDir,htmpath):
    f_qrs = {"Name": {}, "Ref": {}, "test": {}, "Rms error": {}}
    f_p = {"Name": {}, "Ref": {}, "test": {}, "Rms error": {}}
    f_pri = {"Name": {}, "Ref": {}, "test": {}, "Rms error": {}}
    count = 0
    for (dirpath, dirnames, filenames) in walk(inputpath):
        for xname in filenames:
            npath = os.path.join(dirpath, xname)
            fullPath, file_extension = os.path.splitext(npath)
            only_name = os.path.basename(fullPath)
            if file_extension == ".dat":
                if os.path.isfile(fullPath + ".dat") and os.path.isfile(fullPath + ".hea") and os.path.isfile(
                        fullPath + "."+leadname):
                    dataConvert(fullPath, leadname, overallsec, ResultDir)
                    out_1 = generateRefData(fullPath, leadname, ResultDir)
                    flag,p_d,pri_d,qrs_d = compar_tool(htmpath, only_name, fullPath,ResultDir)
                    if flag:
                        f_qrs["Name"][count] =only_name
                        f_qrs["Ref"][count] = qrs_d[0]
                        f_qrs["test"][count] = qrs_d[1]
                        f_qrs["Rms error"][count] =qrs_d[2]

                        f_p["Name"][count] = only_name
                        f_p["Ref"][count] = p_d[0]
                        f_p["test"][count] = p_d[1]
                        f_p["Rms error"][count] = p_d[2]

                        f_pri["Name"][count] = only_name
                        f_pri["Ref"][count] = pri_d[0]
                        f_pri["test"][count] = pri_d[1]
                        f_pri["Rms error"][count] = pri_d[2]

                        count +=1

                    print("Done : ", only_name)
                else:
                    print("file missing : ", only_name)
    createReport(f_qrs, f_p, f_pri)

    return "done"