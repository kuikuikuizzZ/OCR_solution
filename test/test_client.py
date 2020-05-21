import requests
import json

if __name__ == '__main__':
    with open("../sample/invoice_solution/clv2.json") as f:
        solution_dict = json.load(f)
    # print(solution_dict)
    data = dict(solution=json.dumps(solution_dict))
    files = {'image_file': open('../sample/invoiceSample/2.jpg', 'rb')}
    r = requests.post('http://0.0.0.0:9001/api/v1alpha1/inference',
        files=files,data=data)
#     import pdb
#     pdb.set_trace()
    content = json.loads(r.content,encoding='utf-8')
    print(content)
    print('status',r.status_code)
    print('text',r.text)
    r = requests.get('http://0.0.0.0:9001/api/v1alpha1/status',)
    print(r.content)
