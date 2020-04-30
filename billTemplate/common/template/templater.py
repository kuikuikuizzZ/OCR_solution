from .matching import match, RegexMatcher, TextMatcher, computeCombinations
from .findProject import CenterProject, FindProjects
from .mapping import TextBoxesMapping


class TemplateImpl(object):
    def __init__(self, refer_dict, target_dict, regex_dict={}, config=None):
        self.refer_dict = refer_dict
        self.target_dict = target_dict
        self.regex_dict = regex_dict
        self.refRegionDict = {}
        self.refRegionDict.update(refer_dict)
        self.refRegionDict.update(regex_dict)
        # TODO: process adapt to more general
        # matcher
        self.regexMatch = RegexMatcher(regex_dict.keys())
        self.referMatch = TextMatcher(refer_dict.keys())
        # projecter
        if config is not None and config.get('project', None):
            project_cfg = config['project']
            self.project = CenterProject(
                region_type=project_cfg['region_type'],
                pred_type=project_cfg['pred_type'])
            self.findProject = FindProjects(project=self.project)
        else:
            self.project = CenterProject()
            self.findProject = FindProjects(project=self.project)

        # mapper
        self.mapper = TextBoxesMapping()

    def __call__(self, predict_text, predict_boxes):
        '''
        predict_text: a list of string
        predict_boxes: [N,4,2] np.array
        '''
        matchedDict = match([self.regexMatch, self.referMatch], predict_text)
        predcombinations, keyCombinations = computeCombinations(matchedDict)
        result = self.findProject(self.refRegionDict, keyCombinations,
                                  predict_boxes, predcombinations)
        M = result['matrix']
        predResult = (predict_boxes, predict_text)
        mapped_dict = self.mapper(self.target_dict, predResult, M)
        return mapped_dict

    def request():
        ''' request the data'''
        pass
