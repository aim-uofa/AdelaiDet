#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
from collections import namedtuple
from adet.evaluation import rrc_evaluation_funcs
import importlib
import sys

import math 

from rapidfuzz import string_metric

WORD_SPOTTING =True
def evaluation_imports():
    """
    evaluation_imports: Dictionary ( key = module name , value = alias  )  with python modules used in the evaluation. 
    """      
    return {
            'Polygon':'plg',
            'numpy':'np'
            }

def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    global WORD_SPOTTING          
    return {
            'IOU_CONSTRAINT' :0.5,
            'AREA_PRECISION_CONSTRAINT' :0.5,
            'WORD_SPOTTING' :WORD_SPOTTING,
            'MIN_LENGTH_CARE_WORD' :3,
            'GT_SAMPLE_NAME_2_ID':'([0-9]+).txt',
            'DET_SAMPLE_NAME_2_ID':'([0-9]+).txt',            
            'LTRB':False, #LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
            'CRLF':False, # Lines are delimited by Windows CRLF format
            'CONFIDENCES':False, #Detections must include confidence value. MAP and MAR will be calculated,
            'SPECIAL_CHARACTERS':str('!?.:,*"()·[]/\''),
            'ONLY_REMOVE_FIRST_LAST_CHARACTER' : True
        }

def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])
    
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    #Validate format of GroundTruth
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file_gt(k,gt[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True)

    #Validate format of results
    for k in subm:
        if (k in gt) == False :
            raise Exception("The sample %s not present in GT" %k)
        
        rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])

    
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """  
    for module,alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """        
        num_points = len(points)
        # resBoxes=np.empty([1,num_points],dtype='int32')
        resBoxes=np.empty([1,num_points],dtype='float32')
        for inp in range(0, num_points, 2):
            resBoxes[0, int(inp/2)] = float(points[int(inp)])
            resBoxes[0, int(inp/2+num_points/2)] = float(points[int(inp+1)])
        pointMat = resBoxes[0].reshape([2,int(num_points/2)]).T
        return plg.Polygon(pointMat)    

    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin)
        resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin)
        resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax)
        resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax)
        resBoxes[0,7]=int(rect.ymax)

        pointMat = resBoxes[0].reshape([2,4]).T
        
        return plg.Polygon( pointMat)
    
    def rectangle_to_points(rect):
        points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)]
        return points
        
    def get_union(pD,pG):
        areaA = pD.area();
        areaB = pG.area();
        return areaA + areaB - get_intersection(pD, pG);
        
    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG);
        except:
            return 0
        
    def get_intersection(pD,pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()
    
    def compute_ap(confList, matchList,numGtCare):
        correct = 0
        AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)

            if numGtCare>0:
                AP /= numGtCare
            
        return AP  
    
    def transcription_match(transGt,transDet,specialCharacters=str(r'!?.:,*"()·[]/\''),onlyRemoveFirstLastCharacterGT=True):
        
        if onlyRemoveFirstLastCharacterGT:
            #special characters in GT are allowed only at initial or final position
            if (transGt==transDet):
                return True        

            if specialCharacters.find(transGt[0])>-1:
                if transGt[1:]==transDet:
                    return True

            if specialCharacters.find(transGt[-1])>-1:
                if transGt[0:len(transGt)-1]==transDet:
                    return True

            if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
                if transGt[1:len(transGt)-1]==transDet:
                    return True
            return False
        else:
            #Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(transGt)>0 and specialCharacters.find(transGt[0])>-1:
                transGt = transGt[1:]
				
            while len(transDet)>0 and specialCharacters.find(transDet[0])>-1:
                transDet = transDet[1:]
                
            while len(transGt)>0 and specialCharacters.find(transGt[-1])>-1 :
                transGt = transGt[0:len(transGt)-1]
                
            while len(transDet)>0 and specialCharacters.find(transDet[-1])>-1:
                transDet = transDet[0:len(transDet)-1]
                
            return transGt == transDet
                    
    
    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        """        
        #special case 's at final
        if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
            transcription = transcription[0:len(transcription)-2]
        
        #hypens at init or final of the word
        transcription = transcription.strip('-');
        
        specialCharacters = str("'!?.:,*\"()·[]/");
        for character in specialCharacters:
            transcription = transcription.replace(character,' ')
        
        transcription = transcription.strip()
        
        if len(transcription) != len(transcription.replace(" ","")) :
            return False;
        
        if len(transcription) < evaluationParams['MIN_LENGTH_CARE_WORD']:
            return False;
        
        notAllowed = str("×÷·");
        
        range1 = [ ord(u'a'), ord(u'z') ]
        range2 = [ ord(u'A'), ord(u'Z') ]
        range3 = [ ord(u'À'), ord(u'ƿ') ]
        range4 = [ ord(u'Ǆ'), ord(u'ɿ') ]
        range5 = [ ord(u'Ά'), ord(u'Ͽ') ]
        range6 = [ ord(u'-'), ord(u'-') ]
        
        for char in transcription :
            charCode = ord(char)
            if(notAllowed.find(char) != -1):
                return False
            
            valid = ( charCode>=range1[0] and charCode<=range1[1] ) or ( charCode>=range2[0] and charCode<=range2[1] ) or ( charCode>=range3[0] and charCode<=range3[1] ) or ( charCode>=range4[0] and charCode<=range4[1] ) or ( charCode>=range5[0] and charCode<=range5[1] ) or ( charCode>=range6[0] and charCode<=range6[1] )
            if valid == False:
                return False
        
        return True
    
    def include_in_dictionary_transcription(transcription):
        """
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        """
        #special case 's at final
        if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
            transcription = transcription[0:len(transcription)-2]
        
        #hypens at init or final of the word
        transcription = transcription.strip('-');            
        
        specialCharacters = str("'!?.:,*\"()·[]/");
        for character in specialCharacters:
            transcription = transcription.replace(character,' ')
        
        transcription = transcription.strip()
        
        return transcription
    
    perSampleMetrics = {}
    
    matchedSum = 0
    det_only_matchedSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
   
    numGlobalCareGt = 0;
    numGlobalCareDet = 0;
    det_only_numGlobalCareGt = 0;
    det_only_numGlobalCareDet = 0;
   
    arrGlobalConfidences = [];
    arrGlobalMatches = [];

    for resFile in gt:
        # print('resgt', resFile)
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        if (gtFile is None) :
            raise Exception("The file %s is not UTF-8" %resFile)        

        recall = 0
        precision = 0
        hmean = 0    
        detCorrect = 0
        detOnlyCorrect = 0
        iouMat = np.empty([1,1])
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []  
        gtDontCarePolsNum = [] #Array of Ground Truth Polygons' keys marked as don't Care
        det_only_gtDontCarePolsNum = []
        detDontCarePolsNum = [] #Array of Detected Polygons' matched with a don't Care GT
        det_only_detDontCarePolsNum = []
        detMatchedNums = []
        pairs = []
        
        arrSampleConfidences = [];
        arrSampleMatch = [];
        sampleAP = 0;

        pointsList,_,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)

        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            det_only_dontCare = dontCare = transcription == "###" # ctw1500 and total_text gt have been modified to the same format.
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

            #On word spotting we will filter some transcriptions with special characters
            if evaluationParams['WORD_SPOTTING'] :
                if dontCare == False : 
                    if include_in_dictionary(transcription) == False : 
                        dontCare = True
                    else:
                        transcription = include_in_dictionary_transcription(transcription)

            gtTrans.append(transcription)
            if dontCare:
                gtDontCarePolsNum.append( len(gtPols)-1 ) 
            if det_only_dontCare:
                det_only_gtDontCarePolsNum.append( len(gtPols)-1 ) 

        
        if resFile in subm:
            
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile]) 
                    
            pointsList,confidencesList,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents_det(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])
            
            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]
                
                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:                    
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                detTrans.append(transcription)

                if len(gtDontCarePolsNum)>0 :
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break

                if len(det_only_gtDontCarePolsNum)>0 :
                    for dontCarePol in det_only_gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            det_only_detDontCarePolsNum.append( len(detPols)-1 )
                            break
                                 
            
            if len(gtPols)>0 and len(detPols)>0:
                #Calculate IoU and precision matrixs
                outputShape=[len(gtPols),len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols),np.int8)
                detRectMat = np.zeros(len(detPols),np.int8)
                det_only_gtRectMat = np.zeros(len(gtPols),np.int8)
                det_only_detRectMat = np.zeros(len(detPols),np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)
                
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                #detection matched only if transcription is equal
                                # det_only_correct = True
                                # detOnlyCorrect += 1
                                if evaluationParams['WORD_SPOTTING']:
                                    edd = string_metric.levenshtein(gtTrans[gtNum].upper(), detTrans[detNum].upper())
                                    if edd<=0: 
                                        correct = True
                                    else:
                                        correct = False
                                    # correct = gtTrans[gtNum].upper() == detTrans[detNum].upper()
                                else:
                                    try:
                                        correct = transcription_match(gtTrans[gtNum].upper(),detTrans[detNum].upper(),evaluationParams['SPECIAL_CHARACTERS'],evaluationParams['ONLY_REMOVE_FIRST_LAST_CHARACTER'])==True
                                    except: # empty
                                        correct = False
                                detCorrect += (1 if correct else 0)
                                if correct:
                                    detMatchedNums.append(detNum)
                
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if det_only_gtRectMat[gtNum] == 0 and det_only_detRectMat[detNum] == 0 and gtNum not in det_only_gtDontCarePolsNum and detNum not in det_only_detDontCarePolsNum:
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:
                                det_only_gtRectMat[gtNum] = 1
                                det_only_detRectMat[detNum] = 1
                                #detection matched only if transcription is equal
                                det_only_correct = True
                                detOnlyCorrect += 1
                                                              
                
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        det_only_numGtCare = (len(gtPols) - len(det_only_gtDontCarePolsNum))
        det_only_numDetCare = (len(detPols) - len(det_only_detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare >0 else float(1)
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare==0 else float(detCorrect) / numDetCare

        if det_only_numGtCare == 0:
            det_only_recall = float(1)
            det_only_precision = float(0) if det_only_numDetCare >0 else float(1)
        else:
            det_only_recall = float(detOnlyCorrect) / det_only_numGtCare
            det_only_precision = 0 if det_only_numDetCare==0 else float(detOnlyCorrect) / det_only_numDetCare

        
        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
        det_only_hmean = 0 if (det_only_precision + det_only_recall)==0 else 2.0 * det_only_precision * det_only_recall / (det_only_precision + det_only_recall)
            
        matchedSum += detCorrect
        det_only_matchedSum += detOnlyCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        det_only_numGlobalCareGt += det_only_numGtCare
        det_only_numGlobalCareDet += det_only_numDetCare

        perSampleMetrics[resFile] = {
                                        'precision':precision,
                                        'recall':recall,
                                        'hmean':hmean,
                                        'iouMat':[] if len(detPols)>100 else iouMat.tolist(),
                                        'gtPolPoints':gtPolPoints,
                                        'detPolPoints':detPolPoints,
                                        'gtTrans':gtTrans,
                                        'detTrans':detTrans,
                                        'gtDontCare':gtDontCarePolsNum,
                                        'detDontCare':detDontCarePolsNum,
                                        'evaluationParams': evaluationParams,
                                    }
        
    
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    det_only_methodRecall = 0 if det_only_numGlobalCareGt == 0 else float(det_only_matchedSum)/det_only_numGlobalCareGt
    det_only_methodPrecision = 0 if det_only_numGlobalCareDet == 0 else float(det_only_matchedSum)/det_only_numGlobalCareDet
    det_only_methodHmean = 0 if det_only_methodRecall + det_only_methodPrecision==0 else 2* det_only_methodRecall * det_only_methodPrecision / (det_only_methodRecall + det_only_methodPrecision)

    
    methodMetrics = r"E2E_RESULTS: precision: {}, recall: {}, hmean: {}".format(methodPrecision, methodRecall, methodHmean)
    det_only_methodMetrics = r"DETECTION_ONLY_RESULTS: precision: {}, recall: {}, hmean: {}".format(det_only_methodPrecision, det_only_methodRecall, det_only_methodHmean)
    
    
    resDict = {'calculated':True,'Message':'','e2e_method': methodMetrics,'det_only_method': det_only_methodMetrics,'per_sample': perSampleMetrics}
    
    
    return resDict;

def text_eval_main(det_file, gt_file, is_word_spotting):
    global WORD_SPOTTING
    WORD_SPOTTING = is_word_spotting
    return rrc_evaluation_funcs.main_evaluation(None,det_file, gt_file, default_evaluation_params,validate_data,evaluate_method)
