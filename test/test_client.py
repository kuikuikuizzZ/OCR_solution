import requests
import json
import pprint
if __name__ == '__main__':
    with open("../sample/invoice_solution/0.json") as f:
        solution_dict = json.load(f)
    # print(solution_dict)
    data = dict(solution=json.dumps(solution_dict))
    files = {'image_file': open('../sample/invoiceSample/5.jpg', 'rb')}
    r = requests.post('http://0.0.0.0:9001/api/v1alpha1/inference',
        files=files,data=data)
    content = json.loads(r.content,encoding='utf-8')
    pprint.pprint(content)
    
    r = requests.get('http://0.0.0.0:9001/api/v1alpha1/status',)
    content = json.loads(r.content,encoding='utf-8')
    pprint.pprint(content)