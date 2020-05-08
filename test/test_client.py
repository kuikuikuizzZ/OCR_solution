import requests
import json

if __name__ == '__main__':
    with open("../sample/invoice_solution/0.json") as f:
        solution_dict = json.load(f)
    # print(solution_dict)
    data = dict(solution=json.dumps(solution_dict))
    files = {'image_file': open('../sample/invoiceSample/5.jpg', 'rb')}
    r = requests.post('http://10.30.1.3:9010/api/v1alpha1/inference',
        files=files,data=data)
    content = json.loads(r.content,encoding='utf-8')
    print(content)

    r = requests.get('http://10.30.1.3:9010/api/v1alpha1/status',)
    print(r.content)
