import urllib.request
import re
url = 'https://link.springer.com/article/10.3758/s13428-011-0164-y'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = set(re.findall(r'href=[\"\'](.*?\.zip|.*?\.csv|.*?\.xls.*?)[\"\']', html))
    print('Found links:', links)
except Exception as e:
    print('Failed:', e)
