import warnings
import requests

try:
    r=requests.get('https://www.google.com')
except Exception as e:
    pass

try:
    a=1/0
except Exception as e:
    print(e.args[0][:3])

pass
