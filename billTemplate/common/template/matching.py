import re
from copy import deepcopy
from itertools import combinations

def match(matcherList, predictText):
    '''
    matcherList: a list of matcherDict
    '''
    matchDict = dict()
    for matcher in matcherList:
        matcherDict = matcher(predictText)
        if matcherDict.keys() & matchDict.keys():
            raise ValueError('matcher key get conflict.')
        matchDict.update(matcherDict)
    if len(matcherDict) < 4:
        raise ValueError('match points less than 4')
    return matchDict


class Matcher(object):
    def __init__(self, referKey):
        self.referKey = referKey

    def __call__(self, predictText):
        '''
        referKey: a set of key: referRegion label
        predictText: a list of string: each string is a predict content  
        return: matchDict: key: referKey, value: a list of predictText index
        '''
        raise NotImplementedError('matcher __call__ method is not implemented')


class RegexMatcher(Matcher):
    def __call__(self, predictText):
        matchDict = dict()
        for key in self.referKey:
            subPatterns = generatePattern(key, level=len(key) + 1, blank='')
            matchList = matchSubPatterns(subPatterns, predictText)
            if matchList:
                matchDict[key] = matchList
        return matchDict


class TextMatcher(Matcher):
    def __call__(self, predictText):
        matchDict = dict()
        for key in self.referKey:
            subPatterns = generatePattern(key)
            matchList = matchSubPatterns(subPatterns, predictText)
            if matchList:
                matchDict[key] = matchList
        return matchDict


class CharMatcher(Matcher):
    def __call__(self, predictText):
        matchDict = dict()
        for key in self.referKey:
            matchList = matchSubPatterns(key, predictText)
            if matchList:
                matchDict[key] = matchList
        return matchDict

def generatePattern(text, level=3, blank='.*'):
    length = len(text)
    subLength = length - length // level
    #     subtexts = [ item+'?' for item in combinations(text,subLength)]
    subtext_patterns = [ blank+ \
                blank.join(item) + \
                blank for item in combinations(text,subLength)]
    return '|'.join(subtext_patterns)


def matchSubPatterns(subPatterns, predictText, limitMatchNum=2):
    pattern = re.compile(subPatterns)
    matchList = list()
    for i, item in enumerate(predictText):
        match = re.fullmatch(pattern, item)
        if match is not None and len(matchList) < limitMatchNum:
            matchList.append(i)
    return matchList

def computeCombinations(matchedDict):
    '''compute all possible combinations'''
    combinations = [[]]
    keyCombinations = [[]]
    for key, text_list in matchedDict.items():
        if len(text_list) == 1:
            for i in range(len(combinations)):
                combinations[i].append(text_list[0])
                keyCombinations[i].append(key)
        elif len(text_list) > 1:
            combanations_pre = deepcopy(combinations)
            keyCombinations_pre = deepcopy(keyCombinations)
            for i in range(len(combinations)):
                combinations[i].append(text_list[0])
                keyCombinations[i].append(key)
            for text in text_list[1:]:
                new_combanations = deepcopy(combanations_pre)
                new_keyCombinations = deepcopy(keyCombinations_pre)
                for i in range(len(new_combanations)):
                    new_combanations[i].append(text)
                    new_keyCombinations[i].append(key)
                combinations.extend(new_combanations)
                keyCombinations.extend(new_keyCombinations)
    return combinations, keyCombinations
